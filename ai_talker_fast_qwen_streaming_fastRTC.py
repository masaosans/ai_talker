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
#pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/xpu

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

# llama.cpp CUDA
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# audio
pip install numpy librosa soundfile sounddevice

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
import queue

import torch

from faster_whisper import WhisperModel

from llama_cpp import Llama

#from qwen_tts import Qwen3TTSModel
from faster_qwen3_tts import FasterQwen3TTS

from fastrtc import (
    Stream,
    ReplyOnPause,
)

import re
import gc
import threading

from huggingface_hub import hf_hub_download

# ============================================================
# CONFIG
# ============================================================
# dir用意
VOICE_DIR = "./voices"

# HuggingFaceから自動DL
LLM_REPO      = "unsloth/Qwen3.5-2B-GGUF"
LLM_GGUF_FILE = "Qwen3.5-2B-IQ4_XS.gguf"

WHISPER_MODEL = "large-v3-turbo"

TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

N_CTX = 4096

#クローン音声の文字起こし
REFERENCE_TEXT = ""

VOICE_PROMPT = None

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

# TTS モデルへの同時アクセスを防ぐ Lock
tts_lock = threading.Lock()

REFERENCE_AUDIO = None

# ============================================================
# AUDIO OUT (FastRTC経由でブラウザへ)
# ============================================================

# TTSチャンクを FastRTCの response() へ橋渡しするキュー
audio_out_queue = queue.Queue()
# ターン終了センチネル (tts_text_queue用)
TURN_END = object()

# 最後にAIが喋った時刻
LAST_AI_SPEAK_TIME = 0.0

conversation_history = [
    {
        "role": "system",
        "content": """
あなたは優しい男性です。

ルール:
- 短く自然に返答
- 会話テンポ優先
- 長文禁止
- 内部思考を出力しない
- thinkタグを出力しない
- reasoningしない
- 即答する
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
    device="cpu",
    compute_type="int8",
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

print("Downloading/Loading Qwen GGUF from HuggingFace...")

LLM_GGUF = hf_hub_download(
    repo_id=LLM_REPO,
    filename=LLM_GGUF_FILE,
)
print("download complete")
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
# STREAMING LLM
# ============================================================

async def stream_llm(user_text):

    global conversation_history

    conversation_history.append({
        "role": "user",
        "content": user_text
    })

    stream = llm.create_chat_completion(
        messages=conversation_history,
        stream=True,
        temperature=0.7,
        max_tokens=128,
    )

    full_text = ""
    thinking = False

    for chunk in stream:

        if interrupt_event.is_set():
            return

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

async def streaming_tts(text):

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

        # 割り込み済みなら即スキップ
        if interrupt_event.is_set():
            print("[TTS SKIP] interrupted before start")
            return

        # TTS モデルへの排他アクセス
        # 前のターンのTTSがモデルを使い終わるまで待つ
        acquired = await asyncio.to_thread(tts_lock.acquire)
        try:
            if interrupt_event.is_set():
                print("[TTS SKIP] interrupted while waiting lock")
                return

            stream = await asyncio.to_thread(
                tts_model.generate_voice_clone_streaming,
                text=text,
                language="Japanese",
                voice_clone_prompt=VOICE_PROMPT,
                chunk_size=8,
            )
        finally:
            tts_lock.release()

        gen_time = time.perf_counter() - t0
        if gen_time > 0.5:
            print(f"[TTS GEN TIME] {gen_time:.2f}s")

        first_chunk = True
        silent_chunks = 0

        for item in stream:

            if interrupt_event.is_set():
                print("[TTS INTERRUPTED]")
                return

            # streaming版の返却形式
            if isinstance(item, tuple):
                pcm_chunk, sr, timing = item
            elif isinstance(item, dict):
                sr = item.get("sampling_rate", 24000)
                pcm_chunk = item.get("audio", None)
            else:
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
                print(f"[FIRST AUDIO LATENCY] {time.perf_counter()-t0:.3f}s")
                first_chunk = False

            audio_out_queue.put((24000, pcm_chunk))

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

        # ターン終了 → response() のブロックを解除
        if text is TURN_END:
            audio_out_queue.put(None)
            continue

        try:

            await streaming_tts(text)

        except Exception as e:

            print("\n[TTS WORKER ERROR]")
            print(e)

# ============================================================
# PIPELINE
# ============================================================

async def realtime_pipeline(audio):

    global audio_task
    global LAST_AI_SPEAK_TIME

    interrupt_event.clear()

    # 前ターンの残りテキストをクリア（旧TTSテキストを捨てる）
    while not tts_text_queue.empty():
        try:
            tts_text_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    sample_rate, audio_np = audio

    print("\n================================================")
    print("USER")
    print("================================================")

    # ========================================================
    # FastRTC audio normalize
    # ========================================================

    print("\n[AUDIO INFO]")
    print("sample_rate:", sample_rate)
    print("shape:", audio_np.shape)
    print("dtype:", audio_np.dtype)

    # (channels, samples) -> mono
    if audio_np.ndim > 1:

        audio_np = audio_np.mean(axis=0)

    # int16 -> float32 (-1~1)
    if audio_np.dtype == np.int16:

        audio_np = (
            audio_np.astype(np.float32)
            / 32768.0
        )

    else:

        audio_np = audio_np.astype(np.float32)

    # ========================================================
    # AI SPEAKING FILTER
    # ========================================================

    # AI発話直後のマイク入力は捨てる
    # 自分の声ループ防止

    since_ai = time.time() - LAST_AI_SPEAK_TIME

    if since_ai < 0.8:

        print(
            f"\n[SKIP AI VOICE] "
            f"{since_ai:.2f}s"
        )

        audio_out_queue.put(None)
        return

    # ========================================================
    # silence check
    # ========================================================

    peak = np.max(np.abs(audio_np))

    threshold = 0.015

    print("peak:", peak)
    print("threshold:", threshold)

    if peak < threshold:

        print("\n[SKIP] silence\n")
        audio_out_queue.put(None)
        return

    # ========================================================
    # ASR
    # ========================================================

    final_text = ""

    async for partial_text in streaming_asr(
        audio_np,
        sample_rate
    ):

        final_text = partial_text

        print("\n[ASR]")
        print(partial_text)

    if not final_text.strip():

        print("\n[EMPTY ASR]\n")
        audio_out_queue.put(None)
        return

    # ========================================================
    # LLM
    # ========================================================

    token_stream = stream_llm(
        final_text
    )

    async for partial in token_chunker(
        token_stream
    ):

        if interrupt_event.is_set():
            break

        if partial:
            print(f"[AI] {partial}")
            await tts_text_queue.put(partial)

    # ターン終了を通知（中断時も含む）
    await tts_text_queue.put(TURN_END)


# ============================================================
# AUDIO QUEUE CLEAR
# ============================================================

async def clear_audio_queue():

    # キューを空にして response() をアンブロック
    while not audio_out_queue.empty():
        try:
            audio_out_queue.get_nowait()
        except Exception:
            break

    audio_out_queue.put(None)

    print("\n[AUDIO QUEUE CLEARED]\n")

# ============================================================
# CALLBACK
# ============================================================

def response(audio):

    try:

        # 新ターン開始: 常に旧処理を停止してキューをリセット
        interrupt_event.set()

        # audio_out_queue を同期クリア（旧音声を捨てる）
        while not audio_out_queue.empty():
            try:
                audio_out_queue.get_nowait()
            except Exception:
                break

        asyncio.run_coroutine_threadsafe(
            realtime_pipeline(audio),
            loop
        )

    except Exception as e:

        print("\n[PIPELINE ERROR]")
        print(e)
        return

    # TTSチャンクをブラウザへストリーミング送信
    while True:

        try:
            item = audio_out_queue.get(timeout=30)
        except queue.Empty:
            break

        if item is None:
            break

        yield item


# ============================================================
# FASTRTC
# ============================================================

# ============================================================
# TURN SERVER CONFIG
# LAN内 coturn のIPとポートに書き換えてください
# apt install coturn で立てたサーバーのLAN IP
# ============================================================
TURN_SERVER_IP = None
TURN_PORT      = None
TURN_USER      = None
TURN_PASS      = None

stream = Stream(
    ReplyOnPause(
        response,
        can_interrupt=True,
    ),
    modality="audio",
    # AIコンテナを network_mode: host にした場合は rtc_configuration 不要
    # ホストの実IPがICE candidateに入るのでTURN経由不要
)


# ============================================================
# START PLAYBACK WORKER
# ============================================================

loop = asyncio.new_event_loop()

def loop_runner():

    asyncio.set_event_loop(loop)

    loop.create_task(
        tts_worker()
    )


    loop.run_forever()

threading.Thread(
    target=loop_runner,
    daemon=True,
).start()

# ============================================================
# SUPPRESS AIOICE / UVICORN NOISE
# ============================================================

import logging

# uvicorn アクセスログのスパム抑制
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# aioice の send_stun をパッチ
# ブラウザ切断後に socket が None になった状態で
# STUN リトライが来ても AttributeError を静かに無視する
try:
    import aioice.ice as _aioice_ice

    _orig_send_stun = _aioice_ice.StunProtocol.send_stun

    def _safe_send_stun(self, message, addr):
        try:
            _orig_send_stun(self, message, addr)
        except AttributeError:
            pass  # transport/socket 既にクローズ済み → 無視

    _aioice_ice.StunProtocol.send_stun = _safe_send_stun

except Exception:
    pass

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
