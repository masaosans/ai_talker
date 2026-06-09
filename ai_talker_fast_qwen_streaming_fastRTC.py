"""
====================================================================
RTX3080向け 超低遅延リアルタイム音声AI
FastRTC + faster-whisper + Qwen3 4B Q4 + Qwen3-TTS
Voice Clone選択/録音対応 完全版
====================================================================

2026 安定版構成
--------------------------------------------------------------------
✓ faster-whisper へ変更（Qwen3-ASR依存競合回避）
✓ transformers競合回避
✓ RTX3080最適化
✓ Streaming LLM
✓ Streaming TTS
✓ 割り込み対応
✓ Playback Cancel
✓ Voice Clone
✓ Mic Voice Clone
✓ WebRTC
✓ CUDA最適化
✓ chunked realtime response

====================================================================
pip install
====================================================================

# CUDA 12.4
pip uninstall torch torchvision torchaudio
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Transformers
pip install transformers==4.57.3

# 基本
pip install accelerate
pip install sentencepiece

# Qwen3-TTS
#pip install qwen-tts
pip install faster-qwen3-tts

# FastRTC
pip install "fastrtc[vad]"
#vadを利用する実装
pip install silero-vad

# llama.cpp CUDA
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# audio
pip install numpy librosa soundfile #sounddevice

# ASR
pip install faster-whisper

====================================================================
必要モデル
====================================================================

LLM:
Qwen3-4B-Instruct-Q4_K_M.gguf

TTS:
Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

ASR:
faster-whisper large-v3-turbo

====================================================================
Windows追加
====================================================================

ffmpeg:
https://www.gyan.dev/ffmpeg/builds/

sox:
https://sourceforge.net/projects/sox/

====================================================================
"""

import os
import asyncio
import time
from pathlib import Path

# ============================================================
# ENV
# ============================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# HF symlink問題対策
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# 状況把握のlogging用
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ffmpeg / sox
os.environ["PATH"] += r";C:\lib\ffmpeg\bin"
os.environ["PATH"] += r";C:\lib\sox"

# ============================================================
# IMPORT
# ============================================================

import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd

import torch

from faster_whisper import WhisperModel

from llama_cpp import Llama

#from qwen_tts import Qwen3TTSModel
from faster_qwen3_tts import FasterQwen3TTS

#from fastrtc import (
#    Stream,
#    ReplyOnPause,
#)
from fastrtc import Stream, StreamHandler  # StreamHandler も必要
import silero_vad


import re
import gc
import threading



# ============================================================
# CONFIG
# ============================================================

VOICE_DIR = "./voices"

LLM_GGUF = "./models/Qwen3.5-2B-IQ4_XS.gguf"

WHISPER_MODEL = "large-v3-turbo"

TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

N_CTX = 4096

#クローン音声の文字起こし
REFERENCE_TEXT = ""

VOICE_PROMPT = None

global IS_AI_SPERKING
IS_AI_SPERKING = False

os.makedirs(VOICE_DIR, exist_ok=True)

# ============================================================
# CUDA INFO
# ============================================================

print("\n================================================")
print("CUDA INFO")
print("================================================")

print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    vram = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    print(f"VRAM: {vram:.1f} GB")

print("================================================\n")

# ============================================================
# GLOBAL
# ============================================================

interrupt_event = asyncio.Event()

audio_task = None

# queを行う形に修正
tts_text_queue = asyncio.Queue()
#方式変更
#audio_queue = asyncio.Queue()

tts_worker_task = None
playback_worker_task = None

REFERENCE_AUDIO = None

current_pipeline_task = None   # 現在実行中のパイプラインタスク
current_tts_task = None
generator_lock = threading.Lock()

# ============================================================
# RING BUFFER
# ============================================================

RING_BUFFER_SIZE = 24000 * 20

audio_ring = np.zeros(
    RING_BUFFER_SIZE,
    dtype=np.float32
)

ring_write_pos = 0
ring_read_pos = 0

ring_lock = threading.Lock()

# ============================================================
# AUDIO BUFFER SYSTEM
# ============================================================

# 音声を細切れで即再生すると gap が出る
# 一旦 buffer に貯めてから連続再生する

AUDIO_BUFFER_CHUNKS = 2 # 6だと遅すぎる

# 再生中フラグ
IS_AI_SPERKING = False

# 最後にAIが喋った時刻
LAST_AI_SPEAK_TIME = 0.0

MAX_HISTORY_TURNS = 4  # ユーザー＋アシスタントで4往復（8メッセージ）+ system

conversation_history = [
    {
        "role": "system",
        "content": """
あなたは対話型AIです。
楽しく会話を行うことが目的です。

ルール:
- 会話テンポ優先
- 入力メッセージが今までの会話とかみ合わなければ聞き間違いを疑う
- 内部思考を出力しない
- thinkタグを出力しない
"""
    }
]

# ============================================================
# VOICE FILES
# ============================================================

def list_voice_files():

    exts = [
        "*.wav",
        "*.mp3",
        "*.flac"
    ]

    files = []

    for ext in exts:

        files.extend(
            Path(VOICE_DIR).glob(ext)
        )

    return files

# ============================================================
# AUDIO PREPROCESS
# ============================================================

def preprocess_reference_audio(path):

    path = path.strip().strip('"').strip("'")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    print(f"\nLoading voice: {path}")

    audio, sr = librosa.load(
        path,
        sr=24000,
        mono=True
    )

    # silence trim
    audio, _ = librosa.effects.trim(
        audio,
        top_db=20
    )

    # normalize
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    save_path = os.path.join(
        VOICE_DIR,
        f"processed_{int(time.time())}.wav"
    )

    sf.write(
        save_path,
        audio,
        24000
    )

    return save_path

# ============================================================
# RECORD VOICE
# ============================================================

def record_reference_voice():

    print("\n================================================")
    print("Voice Clone録音")
    print("5秒間話してください")
    print("================================================\n")

    samplerate = 24000
    duration = 5

    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    raw_path = os.path.join(
        VOICE_DIR,
        f"mic_voice_{int(time.time())}.wav"
    )

    sf.write(
        raw_path,
        audio,
        samplerate
    )

    processed = preprocess_reference_audio(
        raw_path
    )

    print(f"\n保存: {processed}\n")

    return processed

# ============================================================
# SELECT VOICE
# ============================================================

def select_voice():

    global REFERENCE_AUDIO

    print("""
================================================
Voice Clone設定
================================================

1. 既存voiceを使う
2. マイク録音
3. 外部ファイル指定

================================================
""")

    mode = input("選択: ").strip()

    # ========================================================
    # MIC
    # ========================================================

    if mode == "2":

        REFERENCE_AUDIO = record_reference_voice()
        return

    # ========================================================
    # FILE
    # ========================================================

    elif mode == "3":

        path = input(
            "\n音声ファイルパス: "
        ).strip()

        REFERENCE_AUDIO = preprocess_reference_audio(
            path
        )

        print(f"\n選択: {REFERENCE_AUDIO}\n")

        return

    # ========================================================
    # EXISTING
    # ========================================================

    voices = list_voice_files()

    if len(voices) == 0:

        print("\nvoiceがありません")
        print("録音モードへ移行します\n")

        REFERENCE_AUDIO = record_reference_voice()

        return

    print("\n利用可能voice:\n")

    for idx, file in enumerate(voices):

        print(f"{idx+1}. {file.name}")

    print()

    idx = int(input("番号選択: ")) - 1

    selected = str(voices[idx])

    REFERENCE_AUDIO = selected

    print(f"\n選択: {REFERENCE_AUDIO}\n")

# ============================================================
# SELECT VOICE
# ============================================================

select_voice()

# ============================================================
# LOAD ASR
# ============================================================

print("Loading faster-whisper...")

asr_model = WhisperModel(
    WHISPER_MODEL,
    device="cuda",
    compute_type="float16",
    cpu_threads=4,
    num_workers=1,
)

print("ASR OK")

# ============================================================
# reference音声をASRして参照テキストを作る
# ============================================================

def generate_reference_text(audio_path: str) -> str:

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    audio = audio.astype(np.float32)

    segments, _ = asr_model.transcribe(
        audio,
        language="ja",
        beam_size=1,
        vad_filter=True,
    )

    text = "".join([s.text for s in segments]).strip()

    print("\n================================================")
    print("[REFERENCE TRANSCRIPT]")
    print(text)
    print("================================================\n")

    return text

# ============================================================
# LOAD LLM
# ============================================================

print("Loading Qwen GGUF...")

if not os.path.exists(LLM_GGUF):

    raise FileNotFoundError(
        f"GGUF not found:\n{LLM_GGUF}"
    )

llm = Llama(
    model_path=LLM_GGUF,
    n_gpu_layers=-1,
    n_ctx=N_CTX,
    n_batch=512,
    flash_attn=False,
    use_mmap=True,
    use_mlock=False,
    verbose=False,
    chat_format="chatml"
    
)

print("LLM OK")

# ============================================================
# LOAD TTS
# ============================================================

print("Loading Qwen3-TTS...")

# CUDA断片化抑制
torch.cuda.empty_cache()
gc.collect()

tts_model = FasterQwen3TTS.from_pretrained(
    TTS_MODEL,
)
#    device_map="cuda:0",
#    dtype=torch.bfloat16,
#    attn_implementation="sdpa",



print("TTS OK")

# ============================================================
# （初期処理）参照テキスト作成
# ============================================================

print("TEXT AND PRONPT START")

REFERENCE_TEXT = generate_reference_text(REFERENCE_AUDIO)
print("REFERENCE_TEXT OK")

#プロンプトも生成
VOICE_PROMPT = tts_model.model.create_voice_clone_prompt(
    ref_audio=REFERENCE_AUDIO,
    ref_text=REFERENCE_TEXT,
    x_vector_only_mode=True,
)
print("VOICE_PROMPT OK")

print("TEST RUN")

# 2回目以降の高速化
boo = tts_model.generate_voice_clone(
    text="REFERENCE_TEXT",
    language="Japanese",
    voice_clone_prompt=VOICE_PROMPT,
)
print("TEST RUN OK")

# ============================================================
# DEBUG
# ============================================================

import inspect

print("\n================ TTS METHODS ================\n")

print(dir(tts_model))

print("\n================ SIGNATURE ================\n")

if hasattr(tts_model, "inference_zero_shot"):

    print(
        inspect.signature(
            tts_model.inference_zero_shot
        )
    )

# ============================================================
# STREAMING ASR
# ============================================================

async def streaming_asr(audio_np, sr):

    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    if sr != 16000:

        audio_np = librosa.resample(
            audio_np,
            orig_sr=sr,
            target_sr=16000
        )

    segments, info = asr_model.transcribe(
        audio_np,
        language="ja",
        vad_filter=True,
        beam_size=1,
    )

    text = "".join(
        s.text for s in segments
    )

    yield text.strip()


# ============================================================
# 会話履歴の制御
# ============================================================

def trim_conversation_history():
    global conversation_history
    # system メッセージは先頭に固定
    system_msg = conversation_history[0] if conversation_history and conversation_history[0]["role"] == "system" else None
    # ユーザーとアシスタントのペアを最新の MAX_HISTORY_TURNS ターンだけ残す
    messages = conversation_history if system_msg is None else conversation_history[1:]
    # 最大メッセージ数 = MAX_HISTORY_TURNS * 2
    max_messages = MAX_HISTORY_TURNS * 2
    if len(messages) > max_messages:
        messages = messages[-max_messages:]
        conversation_history = ([system_msg] if system_msg else []) + messages
        print(f"[TRIM] History trimmed to {len(conversation_history)} messages")

# ============================================================
# STREAMING LLM
# ============================================================

async def stream_llm(user_text):

    global conversation_history

    # 会話履歴の削除
    trim_conversation_history()

    conversation_history.append({
        "role": "user",
        "content": user_text
    })

    stream = llm.create_chat_completion(
        messages=conversation_history,
        stream=True,
        temperature=0.7,
        max_tokens=2048,
    )

    full_text = ""
    thinking = False

    for chunk in stream:

        try:

            token = chunk["choices"][0]["delta"].get(
                "content",
                ""
            )

            if not token:
                continue

            # =========================
            # THINK FILTER
            # =========================

            if "<think>" in token:
                thinking = True
                continue

            if "</think>" in token:
                thinking = False
                continue

            if thinking:
                continue

            # special token
            if "<|" in token:
                continue

            full_text += token

            yield token

        except:
            pass

    conversation_history.append({
        "role": "assistant",
        "content": full_text
    })

# ============================================================
# TOKEN CHUNKER
# ============================================================

async def token_chunker(token_stream):

    current = ""

    async for token in token_stream:

        current += token

        # 文末で切る
        if any(
            current.endswith(p)
            #for p in "。、！？.!?\n"
            for p in "。！？.!?\n"
        ):

            yield current.strip()

            current = ""

        # 長すぎ防止
        elif len(current) >= 48:

            yield current.strip()

            current = ""

    if current.strip():

        yield current.strip()

# ============================================================
# STREAMING TTS
# ============================================================
async def _streaming_tts_impl(text):

    try:

        global LAST_AI_SPEAK_TIME

        # ============================================
        # THINK TOKEN REMOVE
        # ============================================

        if not text:
            return

        print("\n[TTS START]")
        print(text)

        t0 = time.perf_counter()

        # ============================================
        # 音声クローン（別スレッド実行）
        # ============================================
        #audio_list, sr  = 
        stream = await asyncio.to_thread(
            tts_model.generate_voice_clone_streaming,
            text=text,
            language="Japanese",
            voice_clone_prompt=VOICE_PROMPT,
            chunk_size=8,
        )


        print(
            f"[TTS GEN TIME] "
            f"{time.perf_counter() - t0:.2f}s"
        )


        first_chunk = True
        first_item = True
        silent_chunks = 0
        
        t1 = time.perf_counter()
        last_chunk_time = time.perf_counter()

        for item in stream:

            now = time.perf_counter()

            print(
                "[CHUNK GAP]",
                now - last_chunk_time
            )

            last_chunk_time = now

            if first_item:
                print(
                    "[FIRST ITEM]",
                    time.perf_counter() - t1
                )
                first_item = False

            # ==================================================
            # streaming版の返却形式
            # ==================================================

            if isinstance(item, tuple):

                print(
                    "[TUPLE TYPES]",
                    type(item[0]),
                    type(item[1])
                )

                # Qwen3-TTS-fast は
                # (audio, sr, timing)
                pcm_chunk, sr, timing  = item

            # dict
            elif isinstance(item, dict):

                sr = item.get("sampling_rate", 24000)

                pcm_chunk = item.get("audio", None)

            else:

                print("[UNKNOWN STREAM ITEM]", item)
                continue

            if pcm_chunk is None:
                continue

            pcm_chunk = np.asarray(
                pcm_chunk,
                dtype=np.float32
            )

            peak = np.max(np.abs(pcm_chunk))


            # ============================================
            # 無音tail除去
            # ============================================

            if peak < 0.002:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks >= 4:
                print("[TTS END DETECT]")
                break

            # stereo -> mono1
            if pcm_chunk.ndim > 1:

                pcm_chunk = pcm_chunk.mean(axis=1)

            pcm_chunk = np.ascontiguousarray(
                pcm_chunk.squeeze()
            )

            if first_chunk:

                print(
                    f"[FIRST AUDIO LATENCY] "
                    f"{time.perf_counter()-t0:.3f}s"
                )

                first_chunk = False

            ring_write(pcm_chunk)

    except Exception as e:

        print("\n[TTS ERROR]")
        print(type(e))
        print(e)
        
# ============================================================
# TTS WORKER
# ============================================================

async def tts_worker():

    print("\n[TTS WORKER STARTED]\n")

    while True:

        text = await tts_text_queue.get()

        if text is None:
            break

        try:

            await streaming_tts(text)

        except Exception as e:

            print("\n[TTS WORKER ERROR]")
            print(e)


# ============================================================
# PIPELINEラッパー
# ============================================================
async def realtime_pipeline(audio):
    global current_pipeline_task
    
    # 前のタスクをキャンセル
    if current_pipeline_task and not current_pipeline_task.done():
        current_pipeline_task.cancel()
        try:
            await current_pipeline_task
        except asyncio.CancelledError:
            print("[PIPELINE] Previous cancelled")
    
    # リングバッファをクリア（再生中の音声を即座に消す）
    with ring_lock:
        global ring_read_pos, ring_write_pos
        ring_read_pos = 0
        ring_write_pos = 0
        audio_ring.fill(0)
    
    # 新しいパイプラインを実行
    current_pipeline_task = asyncio.create_task(_realtime_pipeline_impl(audio))
    try:
        await current_pipeline_task
    except asyncio.CancelledError:
        print("[PIPELINE] Current cancelled")
    finally:
        if current_pipeline_task == asyncio.current_task():
            current_pipeline_task = None

# ============================================================
# PIPELINE
# ============================================================

async def _realtime_pipeline_impl(audio):
    global LAST_AI_SPEAK_TIME

    sample_rate, audio_np = audio

    print("\n================================================")
    print("USER")
    print("================================================")

    print("\n[AUDIO INFO]")
    print("sample_rate:", sample_rate)
    print("shape:", audio_np.shape)
    print("dtype:", audio_np.dtype)

    # (channels, samples) -> mono
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)

    # int16 -> float32 (-1~1)
    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768.0
    else:
        audio_np = audio_np.astype(np.float32)

    # AI発話直後のマイク入力は捨てる（クールダウン 1.5秒に延長）
    since_ai = time.time() - LAST_AI_SPEAK_TIME
    if since_ai < 1.5:
        print(f"\n[SKIP AI VOICE] {since_ai:.2f}s")
        return

    # 無音チェック
    peak = np.max(np.abs(audio_np))
    threshold = 0.015
    print("peak:", peak, "threshold:", threshold)
    if peak < threshold:
        print("\n[SKIP] silence\n")
        return

    # ASR
    final_text = ""
    async for partial_text in streaming_asr(audio_np, sample_rate):
        final_text = partial_text
        print("\n[ASR]", partial_text)

    if not final_text.strip():
        print("\n[EMPTY ASR]\n")
        return

    # LLM
    token_stream = stream_llm(final_text)
    async for partial in token_chunker(token_stream):
        print("\n[AI]", partial)
        await tts_text_queue.put(partial)

# ------------------------------------------------------------
# TTS のタスク管理ラッパー
# ------------------------------------------------------------
async def streaming_tts(text):
    global current_tts_task
    if current_tts_task and not current_tts_task.done():
        current_tts_task.cancel()
        try:
            await current_tts_task
        except asyncio.CancelledError:
            print("[TTS] Previous task cancelled")
    current_tts_task = asyncio.create_task(_streaming_tts_impl(text))
    try:
        await current_tts_task
    except asyncio.CancelledError:
        print("[TTS] Current task cancelled")
    finally:
        if current_tts_task == asyncio.current_task():
            current_tts_task = None

# ============================================================
# ring_available
# ============================================================

def ring_available():

    with ring_lock:

        return (
            ring_write_pos
            - ring_read_pos
        ) % RING_BUFFER_SIZE

# ============================================================
# ring_read
# ============================================================

def ring_read(frames):

    global ring_read_pos

    out = np.zeros(
        frames,
        dtype=np.float32
    )

    with ring_lock:

        available = (
            ring_write_pos
            - ring_read_pos
        ) % RING_BUFFER_SIZE

        if available == 0:
            return out

        read_frames = min(
            frames,
            available
        )

        end = ring_read_pos + read_frames

        if end < RING_BUFFER_SIZE:

            out[:read_frames] = (
                audio_ring[
                    ring_read_pos:end
                ]
            )

        else:

            first = (
                RING_BUFFER_SIZE
                - ring_read_pos
            )

            out[:first] = (
                audio_ring[
                    ring_read_pos:
                ]
            )

            out[first:read_frames] = (
                audio_ring[
                    :read_frames-first
                ]
            )

        ring_read_pos = (
            ring_read_pos + read_frames
        ) % RING_BUFFER_SIZE

    return out


# ============================================================
# ring write
# ============================================================

def ring_write(audio):

    global ring_write_pos, LAST_AI_SPEAK_TIME   # ← LAST_AI_SPEAK_TIME を追加

    audio = np.asarray(
        audio,
        dtype=np.float32
    ).reshape(-1)

    n = len(audio)

    with ring_lock:

        end = ring_write_pos + n

        # wrapなし
        if end < RING_BUFFER_SIZE:

            audio_ring[
                ring_write_pos:end
            ] = audio

        # wrapあり
        else:

            first = (
                RING_BUFFER_SIZE
                - ring_write_pos
            )

            audio_ring[
                ring_write_pos:
            ] = audio[:first]

            audio_ring[
                :n-first
            ] = audio[first:]

        ring_write_pos = (
            ring_write_pos + n
        ) % RING_BUFFER_SIZE

        # 書き込みが完了したら時刻を更新（AIが喋っている最中であることを記録）
        LAST_AI_SPEAK_TIME = time.time()


# ============================================================
# AUDIO QUEUE CLEAR
# ============================================================

async def clear_audio_queue():

    global ring_read_pos
    global ring_write_pos

    with ring_lock:

        ring_read_pos = 0
        ring_write_pos = 0

        audio_ring.fill(0)

    print("\n[RING BUFFER CLEARED]\n")

# ============================================================
# CALLBACK
# ============================================================

class VoiceAIHandler(StreamHandler):
    def __init__(self):
        super().__init__(input_sample_rate=48000, output_sample_rate=24000)
        # Silero VAD モデルをロード（ONNX or Torch）
        self.vad_model = silero_vad.load_silero_vad()
        # 設定
        self.sample_rate = 48000
        self.chunk_duration = 0.5  # 発話判定の基本単位（秒）
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.silence_timeout = 0.6  # 無音が続くべき秒数
        self.silence_chunks_needed = int(self.silence_timeout / self.chunk_duration)
        
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.speech_buffer = None   # 発話中のバッファ
        self.silence_counter = 0
        self.current_task = None

    def _is_speech(self, audio_chunk):
        """音声チャンクが音声を含むかどうかをVADで判定"""
        # silero_vad は 16kHz を想定しているためリサンプル必要
        if self.sample_rate != 16000:
            # librosa などでリサンプル（簡易的にダウンサンプリング）
            import librosa
            chunk_16k = librosa.resample(audio_chunk, orig_sr=self.sample_rate, target_sr=16000)
        else:
            chunk_16k = audio_chunk
        # テンソルに変換してVAD実行
        import torch
        tensor = torch.from_numpy(chunk_16k).float()
        speech_prob = silero_vad.get_speech_timestamps(tensor, self.vad_model, sampling_rate=16000)
        return len(speech_prob) > 0

    def receive(self, frame):
        sr, audio_np = frame
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        
        with self.buffer_lock:
            # 常にバッファに追加
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_np))
            
            # 設定したチャンクサイズ未満なら何もしない
            if len(self.audio_buffer) < self.chunk_samples:
                return
            
            # チャンクを切り出し
            chunk = self.audio_buffer[:self.chunk_samples]
            self.audio_buffer = self.audio_buffer[self.chunk_samples:]
            
            # VAD で音声判定
            if self._is_speech(chunk):
                # 音声あり
                self.silence_counter = 0
                if self.speech_buffer is None:
                    self.speech_buffer = chunk
                else:
                    self.speech_buffer = np.concatenate((self.speech_buffer, chunk))
            else:
                # 無音
                self.silence_counter += 1
                if self.speech_buffer is not None:
                    # 無音が閾値に達したら発話終了
                    if self.silence_counter >= self.silence_chunks_needed:
                        # 発話終了！ 溜めた音声を処理
                        full_audio = self.speech_buffer
                        self.speech_buffer = None
                        asyncio.run_coroutine_threadsafe(
                            self._process_audio(full_audio, sr),
                            background_loop
                        )
                    else:
                        # まだ発話継続中とみなし、無音チャンクもバッファに追加
                        if self.speech_buffer is not None:
                            self.speech_buffer = np.concatenate((self.speech_buffer, chunk))

    async def _process_audio(self, audio_np, original_sr):
        # 前のタスクをキャンセル
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass
            # リングバッファクリア
            with ring_lock:
                global ring_read_pos, ring_write_pos
                ring_read_pos = 0
                ring_write_pos = 0
                audio_ring.fill(0)
        
        # 新しいパイプライン開始
        self.current_task = asyncio.create_task(
            _realtime_pipeline_impl((original_sr, audio_np))
        )
        try:
            await self.current_task
        except asyncio.CancelledError:
            pass
        finally:
            if self.current_task == asyncio.current_task():
                self.current_task = None

    def emit(self):
        chunk_size = 960
        available = ring_available()
        if available < chunk_size:
            return None
        chunk = ring_read(chunk_size)
        return (24000, chunk)

    def copy(self):
        return VoiceAIHandler()

# ============================================================
# FASTRTC
# ============================================================
# Stream の作成
stream = Stream(VoiceAIHandler(), modality="audio")

# ============================================================
# START PLAYBACK WORKER
# ============================================================
background_loop = asyncio.new_event_loop()
def loop_runner():
    asyncio.set_event_loop(background_loop)
    background_loop.create_task(tts_worker())
    background_loop.run_forever()
threading.Thread(target=loop_runner, daemon=True).start()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("""
====================================================================
Realtime Voice AI
====================================================================

Browser:
http://127.0.0.1:7860

====================================================================
""")

    stream.ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
