"""
====================================================================
リアルタイム音声AI
====================================================================

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

#絵文字トリム用
pip install emoji

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
#from fastrtc import Stream, AsyncStreamHandler
from fastrtc import Stream, StreamHandler  # StreamHandler も必要
import silero_vad


import re
import gc
import threading

import emoji


# ============================================================
# UI追加
# ============================================================
import gradio as gr
import uvicorn
from fastapi import FastAPI

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
# 音声感度設定
# ============================================================
#AI発話監視感覚
CHECK_INTERVAL = 0.5

#AI最終発話後のクールタイム（エコー対策）
COOL_TIME = 0.7

#マイクの感度（AI発話中には感度低く）
MIC_THRESHOLD =  0.4
MIC_THRESHOLD_AI_SPERK = 0.6

#割り込み判定値（AI発話中はAI音声の可能性を踏まえ厳しめに）
CHECK_THRESHOLD = 0.015
CHECK_THRESHOLD_AI_SPERK = 0.4

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

#LLMの会話履歴保持数
MAX_HISTORY_TURNS = 3  # ユーザー＋アシスタントで3往復（6メッセージ）+ system

#LLMのプロンプト
conversation_history = [
    {
        "role": "system",
        "content": """
あなたは対話型AIです。ルールを厳守し、応答してください。

# ルール:
- あなたは音声読み上げ前提で応答する。箇条書き禁止。絵文字、記号禁止。読上可能な文章で応答する。
- キャラクター設定になりきって回答する。
- 過剰な長文、短文での応対は禁止。

# キャラクター設定:
- 優しい男性。娘に話しかけるように。
- 難しい言葉を避け、丁寧に応対する。

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
    #chat_format="chatml" #無限ループEOS欠落の原因？
    offload_kqv=True,   # KVキャッシュのGPUオフロードを有効化
    logits_all=True,    # ロジット計算の安定化を図る

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

print("TTS OK")

# ============================================================
# （初期処理）参照テキスト作成
# ============================================================

print("TEXT AND PRONPT START")

# ============================================================
# 初期設定（UIで設定されるまでデフォルトの音声を使う）
# ============================================================
# デフォルトの音声ファイルがあれば自動設定
default_voices = list_voice_files()
print("[DEBUG] Voice files at startup:", list_voice_files())  

if default_voices:
    REFERENCE_AUDIO = str(default_voices[0])
    REFERENCE_TEXT = generate_reference_text(REFERENCE_AUDIO)
    VOICE_PROMPT = tts_model.model.create_voice_clone_prompt(
        ref_audio=REFERENCE_AUDIO,
        ref_text=REFERENCE_TEXT,
        x_vector_only_mode=True,
    )
    print(f"[INIT] Default voice: {os.path.basename(REFERENCE_AUDIO)}")

    print("TEST RUN")

    # 2回目以降の高速化
    boo = tts_model.generate_voice_clone(
        text=REFERENCE_TEXT,
        language="Japanese",
        voice_clone_prompt=VOICE_PROMPT,
    )
    print("TEST RUN OK")

else:
    print("[WARN] No voice file found. Please set voice from browser UI.")



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
    print("\n[TRIM HISTORY]\n")

    # 会話履歴の削除
    trim_conversation_history()

    conversation_history.append({
        "role": "user",
        "content": user_text
    })
    print("\n[HISTORY]:" , conversation_history)
    stream = llm.create_chat_completion(
        messages=conversation_history,
        stream=True,
        temperature=0.7,
        max_tokens=2048,
    )

    print("\n[GET STREAM]\n")

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

        except Exception as e:
            print("\n[AI LLM ERROR]")
            print(type(e))
            print(e)
            pass #継続する

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

        # 長すぎ防止→記号などだけの文字列入力が生まれるため除去！
        #elif len(current) >= 48:
            #yield current.strip()
            #current = ""

    if current.strip():

        yield current.strip()


# ============================================================
# 絵文字での異常動作対策"""文字列から絵文字（BMP外の文字）を除去する"""
# ============================================================
def remove_emojis(text: str) -> str:
    # 受け取ったテキストから絵文字をすべて削除する
    return emoji.replace_emoji(text, replace='')
# ============================================================
# STREAMING TTS
# ============================================================
async def _streaming_tts_impl(text):

    try:
        #print(f"\n[text] original: {repr(text)}")
        text = remove_emojis(text)
        #print(f"[text] after remove_emojis: {repr(text)}")
        text = text.strip()
        #print(f"[text] after strip: {repr(text)}")

        if not text:
            print("\n[TTS NO TEXT]",text)
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
            await asyncio.sleep(0)   # イベントループに制御を戻す(待ち処理に一度制御渡す)

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

    sample_rate, audio_np = audio

    print("\n================================================")
    print("USER")
    print("================================================")

    print("\n[AUDIO INFO]")
    print("sample_rate:", sample_rate)
    print("shape:", audio_np.shape)
    print("dtype:", audio_np.dtype)

    # ASR
    final_text = ""
    async for partial_text in streaming_asr(audio_np, sample_rate):
        final_text = partial_text
        print("\n[ASR]", partial_text)

    if not final_text.strip():
        print("\n[EMPTY ASR]\n")
        return

    # LLM
    print("\n[START LLM]\n")
    token_stream = stream_llm(final_text)
    async for partial in token_chunker(token_stream):
        print("\n[LLM]", partial)
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

    #発話制御は監視側に完全移行
    #global ring_write_pos, LAST_AI_SPEAK_TIME   # ← LAST_AI_SPEAK_TIME を追加
    global ring_write_pos

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
        #LAST_AI_SPEAK_TIME = time.time()


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
    """
    FastRTC用音声ハンドラ（割り込み機能付き完全版）

    処理フロー:
    1. receive() で音声チャンクを受信
    2. 元のコードの音量チェック・クールダウン・ピークチェックを実施
    3. 条件を満たせば _interrupt() を呼び出し（暫定割り込み）
    4. 発話終了後 _process_audio() で再度チェック
    5. 有効ならパイプライン起動、無効ならスキップ
    """

    def __init__(self):
        super().__init__(input_sample_rate=48000, output_sample_rate=24000)

        # ---- VAD設定 ----
        self.vad_model = silero_vad.load_silero_vad()
        self.sample_rate = 48000
        self.chunk_duration = 0.5
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.silence_timeout = 0.6
        self.silence_chunks_needed = int(self.silence_timeout / self.chunk_duration)

        # ---- バッファ ----
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.speech_buffer = None
        self.silence_counter = 0
        self.current_task = None

        # ---- 内部キュー監視（LAST_AI_SPEAK_TIME更新用） ----
        self._output_queue = None
        self._last_check_time = 0
        self._check_interval = CHECK_INTERVAL
        self._is_check = False

    # ============================================================
    # 内部キューアクセス
    # ============================================================
    def _get_output_queue(self):
        """FastRTC内部の音声出力キューを取得する（キャッシュ対応）"""
        if self._output_queue is not None:
            return self._output_queue
        try:
            if hasattr(self, '_clear_queue') and hasattr(self._clear_queue, '__self__'):
                audio_callback = self._clear_queue.__self__
                if hasattr(audio_callback, 'queue'):
                    self._output_queue = audio_callback.queue
                    return self._output_queue
        except Exception as e:
            print(f"[INTERNAL] Failed to access queue: {e}")
        return None

    # ============================================================
    # 出力キュー監視
    # ============================================================
    def _check_output_queue(self):
        """内部キューをチェックし、LAST_AI_SPEAK_TIME を更新"""
        global LAST_AI_SPEAK_TIME
        queue = self._get_output_queue()
        if queue is not None:
            try:
                qsize = queue.qsize()
                if qsize > 0:
                    LAST_AI_SPEAK_TIME = time.time()
                else:
                    self._is_check = False
                    LAST_AI_SPEAK_TIME = time.time()
            except Exception as e:
                print(f"[CHECK] Error: {e}")

    # ============================================================
    # VAD（音声活動検出）
    # ============================================================
    def _is_speech(self, audio_chunk):
        """チャンクに音声が含まれるか Silero VAD で判定"""
        global LAST_AI_SPEAK_TIME

        if self.sample_rate != 16000:
            import librosa
            chunk_16k = librosa.resample(audio_chunk, orig_sr=self.sample_rate, target_sr=16000)
        else:
            chunk_16k = audio_chunk

        #感度調整：AIのノイズをしゃべり続けていると判断させず、マイクインプットを適切に切り取るための設定
        threshold = MIC_THRESHOLD
        since_ai = time.time() - LAST_AI_SPEAK_TIME
        if since_ai < COOL_TIME:
            threshold = MIC_THRESHOLD_AI_SPERK


        import torch
        tensor = torch.from_numpy(chunk_16k).float()
        speech_prob = silero_vad.get_speech_timestamps(
            tensor,
            self.vad_model,
            sampling_rate=16000,
            threshold=threshold
        )
        return len(speech_prob) > 0

    # ============================================================
    # 割り込み処理（新規追加）
    # ============================================================
    def _interrupt(self):
        """
        新規発話検出時に前のAI音声を即座に停止する。
        - タスクキャンセル
        - リングバッファクリア
        - FastRTC出力キューをクリア（clear_queue()）
        - TTSテキストキューをクリア
        """
        global current_pipeline_task, current_tts_task, ring_read_pos, ring_write_pos

        # 1. タスクキャンセル
        for task in (current_pipeline_task, current_tts_task):
            if task and not task.done():
                task.cancel()

        # 2. リングバッファクリア
        with ring_lock:
            ring_read_pos = 0
            ring_write_pos = 0
            audio_ring.fill(0)

        # 3. FastRTC出力キューをクリア
        self.clear_queue()

        # 4. TTSテキストキューをクリア
        while not tts_text_queue.empty():
            try:
                tts_text_queue.get_nowait()
            except:
                break

    # ============================================================
    # 音声チェック（共通関数化）
    # ============================================================
    def _check_audio(self, audio_np):
        """
        元のコードの音量チェック・クールダウン・ピークチェックを実施。
        戻り値: (peak, threshold, is_valid)
        """

        # しきい値設定
        threshold = CHECK_THRESHOLD
        since_ai = time.time() - LAST_AI_SPEAK_TIME
        if since_ai < COOL_TIME:
            threshold = CHECK_THRESHOLD_AI_SPERK

        # ピーク計算
        peak = np.max(np.abs(audio_np))

        return peak, threshold, peak >= threshold

    # ============================================================
    # FastRTC コールバック：入力音声フレームを受信
    # ============================================================
    def receive(self, frame):
        sr, audio_np = frame

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0

        # ---- バッファに蓄積 ----
        with self.buffer_lock:
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_np))
            if len(self.audio_buffer) < self.chunk_samples:
                return

            chunk = self.audio_buffer[:self.chunk_samples]
            self.audio_buffer = self.audio_buffer[self.chunk_samples:]

            # ---- VAD判定 ----
            if self._is_speech(chunk):
                self.silence_counter = 0
                if self.speech_buffer is None:
                    self.speech_buffer = chunk
                else:
                    self.speech_buffer = np.concatenate((self.speech_buffer, chunk))
            else:
                # 無音処理
                self.silence_counter += 1
                if self.speech_buffer is not None:
                    if self.silence_counter >= self.silence_chunks_needed:
                        print("[speech buffer stop]")
                        full_audio = self.speech_buffer
                        self.speech_buffer = None
                        asyncio.run_coroutine_threadsafe(
                            self._process_audio(full_audio, sr),
                            background_loop
                        )
                    else:
                        self.speech_buffer = np.concatenate((self.speech_buffer, chunk))

    # ============================================================
    # 発話音声をモノラル化、正規化後にパイプラインに渡す
    # ============================================================
    async def _process_audio(self, audio_np, original_sr):

        # 前処理（モノラル化・正規化）
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)
        audio_np = audio_np.astype(np.float32)
        max_abs = np.abs(audio_np).max()
        if max_abs > 1.0:
            audio_np = audio_np / 32768.0

        # 音声チェック：集めた音声自体がノイズの場合はドロップするため
        peak, threshold, is_valid = self._check_audio(audio_np)

        # 無効な音声はバッファにも入れずに破棄
        if not is_valid:
            print("\n[SKIP] silence.[peak]",peak,"[threshold]",threshold)
            return

        # ★ 新規発話検出 → 割り込み実行 ★
        print("\nSTART PIPELINE.[peak]",peak,"[threshold]",threshold)
        self._interrupt()

        # ---- パイプライン起動 ----
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass

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

    # ============================================================
    # FastRTC コールバック：出力音声フレームを要求
    # ============================================================
    def emit(self):
        now = time.time()
        #定期感覚で発話チェックを呼び出す
        if now - self._last_check_time >= self._check_interval:
            self._last_check_time = now
            if self._is_check:
                self._check_output_queue()

        chunk_size = 960
        available = ring_available()
        if available < chunk_size:
            return None

        #発話があると確認されたらチェック対象とするフラグを立てる
        self._is_check = True
        chunk = ring_read(chunk_size)
        return (24000, chunk)

    # ============================================================
    # FastRTC 必須：ハンドラのコピーを作成
    # ============================================================
    def copy(self):
        return VoiceAIHandler()

# ============================================================
# FASTRTC
# ============================================================
# Stream の作成
stream = Stream(VoiceAIHandler(), modality="audio")
print("[DEBUG] stream.ui:", stream.ui)          # ← 追加
print("[DEBUG] stream.ui type:", type(stream.ui)) # ← 追加

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
# GRADIO UI（ブラウザ操作用）
# ============================================================

# ---- UIハンドラ関数 ----
def process_voice_from_mic(audio_data):
    """マイク録音（numpy）から音声クローンを設定"""
    global REFERENCE_AUDIO, REFERENCE_TEXT, VOICE_PROMPT
    
    if audio_data is None:
        return gr.update(), "❌ 音声が取得できませんでした"
    
    sr, audio_np = audio_data
    temp_path = f"temp_mic_{int(time.time())}.wav"
    sf.write(temp_path, audio_np, sr)
    
    processed = preprocess_reference_audio(temp_path)
    
    REFERENCE_AUDIO = processed
    REFERENCE_TEXT = generate_reference_text(processed)
    VOICE_PROMPT = tts_model.model.create_voice_clone_prompt(
        ref_audio=processed,
        ref_text=REFERENCE_TEXT,
        x_vector_only_mode=True,
    )
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # プルダウン更新用の戻り値
    updated_choices = refresh_voice_list()
    selected_name = os.path.basename(processed)
    return gr.update(choices=updated_choices, value=selected_name), f"✅ マイク音声で設定完了: {selected_name}"

def process_voice_from_file(file_path):
    """アップロードファイルから音声クローンを設定"""
    global REFERENCE_AUDIO, REFERENCE_TEXT, VOICE_PROMPT
    
    if file_path is None:
        return gr.update(), "❌ ファイルが選択されていません"
    
    processed = preprocess_reference_audio(file_path)
    REFERENCE_AUDIO = processed
    REFERENCE_TEXT = generate_reference_text(processed)
    VOICE_PROMPT = tts_model.model.create_voice_clone_prompt(
        ref_audio=processed,
        ref_text=REFERENCE_TEXT,
        x_vector_only_mode=True,
    )
    
    # プルダウン更新用の戻り値
    updated_choices = refresh_voice_list()
    selected_name = os.path.basename(processed)
    return gr.update(choices=updated_choices, value=selected_name), f"✅ ファイルで設定完了: {selected_name}"

def refresh_voice_list():
    """既存音声一覧を更新"""
    voices = list_voice_files()
    return [f.name for f in voices]

def select_existing_voice(voice_name):
    """ドロップダウンで選択した音声を設定"""
    global REFERENCE_AUDIO, REFERENCE_TEXT, VOICE_PROMPT
    
    if not voice_name:
        return "❌ 音声が選択されていません"
    
    voice_path = os.path.join(VOICE_DIR, voice_name)
    if not os.path.exists(voice_path):
        return f"❌ ファイルが見つかりません: {voice_name}"
    
    #processed = preprocess_reference_audio(voice_path)
    processed = voice_path #ファイルの再生性はしない
    REFERENCE_AUDIO = processed
    REFERENCE_TEXT = generate_reference_text(processed)
    VOICE_PROMPT = tts_model.model.create_voice_clone_prompt(
        ref_audio=processed,
        ref_text=REFERENCE_TEXT,
        x_vector_only_mode=True,
    )
    
    return f"✅ 選択完了: {voice_name}"

# ---- Gradio UI ----
# ---- stream.ui にカスタムコンポーネントを追加 ----
with stream.ui:
    #gr.Markdown("## 🎤 音声クローン設定")
    # ---- アコーディオンで全体を囲む ----
    with gr.Accordion("🎤 音声クローン設定(最初に設定)", open=False):
        with gr.Row():
            mic_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="マイクで録音（5秒）",
                interactive=True,
                value=None
            )

        with gr.Row():
            voice_dropdown = gr.Dropdown(
                choices=refresh_voice_list(),
                label="既存の音声を選択",
                interactive=True,
                value=None
            )
            refresh_btn = gr.Button("🔄 更新")
            select_btn = gr.Button("✅ この音声を使う")

        with gr.Row():
            file_input = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="音声ファイルをアップロード",
                interactive=True,
                value=None
            )


        status = gr.Textbox(
            label="ステータス",
            interactive=False,
            lines=3,
            value="🟡 待機中（WebRTC接続後、話しかけてください）"
        )

        # ---- イベント接続 ----
        mic_input.change(
            fn=process_voice_from_mic,
            inputs=mic_input,
            outputs=[voice_dropdown, status]
        )
        file_input.change(
            fn=process_voice_from_file,
            inputs=file_input,
            outputs=[voice_dropdown, status]
        )
        refresh_btn.click(
            fn=refresh_voice_list,
            outputs=voice_dropdown
        )
        select_btn.click(
            fn=select_existing_voice,
            inputs=voice_dropdown,
            outputs=status
        )

# ---- FastAPIアプリ作成 ----
app = FastAPI()
stream.mount(app)  # WebRTCエンドポイント追加

# ---- FastRTCのUI（カスタムコンポーネント追加済み）を / にマウント ----
app = gr.mount_gradio_app(app, stream.ui, path="/")

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

    #stream.ui.launch(server_name="0.0.0.0",server_port=7860,share=False,)

    # 元の stream.ui.launch() を削除し、代わりに uvicorn で起動
    uvicorn.run(app, host="0.0.0.0", port=7860)
