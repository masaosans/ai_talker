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
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Transformers
pip install transformers==4.57.3

# 基本
pip install accelerate
pip install sentencepiece

# Qwen3-TTS
pip install qwen-tts

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
import sounddevice as sd

import torch

from faster_whisper import WhisperModel

from llama_cpp import Llama

from qwen_tts import Qwen3TTSModel

from fastrtc import (
    Stream,
    ReplyOnPause,
)

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
#queを行う形に修正
tts_text_queue = asyncio.Queue()
audio_queue = asyncio.Queue()

tts_worker_task = None
playback_worker_task = None

REFERENCE_AUDIO = None

conversation_history = [
    {
        "role": "system",
        "content": """
あなたはリアルタイム音声AIです。

- 短く自然に返答
- 会話テンポ優先
- 長文禁止
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

    REFERENCE_AUDIO = preprocess_reference_audio(
        selected
    )

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

tts_model = Qwen3TTSModel.from_pretrained(
    TTS_MODEL,
    device_map="cuda:0",
    dtype=torch.float32,
)

# 推論モード
tts_model.model.eval()

# gradient無効
for p in tts_model.model.parameters():
    p.requires_grad = False

print("TTS OK")

# ============================================================
# （初期処理）参照テキスト作成
# ============================================================

print("TEXT AND PRONPT START")

REFERENCE_TEXT = generate_reference_text(REFERENCE_AUDIO)
print("REFERENCE_TEXT OK")

#プロンプトも生成
VOICE_PROMPT = tts_model.create_voice_clone_prompt(
    ref_audio=REFERENCE_AUDIO,
    ref_text=REFERENCE_TEXT,
)
print("VOICE_PROMPT OK")


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

    for chunk in stream:

        if interrupt_event.is_set():
            return

        try:

            token = chunk["choices"][0]["delta"].get(
                "content",
                ""
            )


            if token:

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

    punctuation = [
        "。",
        "！",
        "？",
        ".",
        "!",
        "?",
        "、",
        ","
    ]

    async for token in token_stream:

        current += token

        if (
            len(current) >= 12
            or any(current.endswith(p) for p in punctuation)
        ):

            yield current

            current = ""

    if current:
        yield current

# ============================================================
# STREAMING TTS
# ============================================================

async def streaming_tts(text):

    try:

        # ============================================
        # THINK TOKEN REMOVE
        # ============================================

        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL
        )

        # special token cleanup
        text = re.sub(r"<\|.*?\|>", "", text)

        text = text.strip()

        if not text:
            return

        # ============================================
        # 音声クローン
        # ============================================

        result = tts_model.generate_voice_clone(
            text=text,
            language="Japanese",
            voice_clone_prompt=VOICE_PROMPT,
            #non_streaming_mode=True,
        )

        print("\n[TTS RAW TYPE]")
        print(type(result))

        # ====================================================
        # various return formats
        # ====================================================

        if isinstance(result, tuple):

            audio_np = result[0]

        elif isinstance(result, dict):

            if "audio" in result:

                audio_np = result["audio"]

            elif "wav" in result:

                audio_np = result["wav"]

            else:

                audio_np = list(result.values())[0]

        else:

            audio_np = result

        # ====================================================
        # numpy
        # ====================================================

        audio_np = np.asarray(
            audio_np,
            dtype=np.float32
        )

        print("\n[TTS AUDIO]")
        print("shape:", audio_np.shape)
        print("dtype:", audio_np.dtype)

        # ====================================================
        # empty safety
        # ====================================================

        if audio_np.size == 0:

            print("\n[TTS EMPTY]")
            return

        # ====================================================
        # stereo -> mono
        # ====================================================

        if audio_np.ndim == 2:

            if audio_np.shape[0] <= 2:

                audio_np = audio_np.mean(axis=0)

            else:

                audio_np = audio_np.mean(axis=1)

        # ====================================================
        # normalize
        # ====================================================

        peak = np.max(np.abs(audio_np))

        print("peak:", peak)

        if peak > 0:

            audio_np = audio_np / peak

        # ====================================================
        # contiguous
        # ====================================================

        audio_np = np.ascontiguousarray(
            audio_np.astype(np.float32)
        )

        # ====================================================
        # streaming
        # ====================================================

        #chunk_size = 8192
        chunk_size = 2048

        for i in range(
            0,
            len(audio_np),
            chunk_size
        ):

            if interrupt_event.is_set():

                print("\n[TTS INTERRUPTED]")
                return

            #yield audio_np[
            #    i:i + chunk_size
            #]
            # yieldしない
            await audio_queue.put(
               audio_np[i:i + chunk_size]
)

            await asyncio.sleep(0)

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
# AUDIO PLAYBACK WORKER
# ============================================================

async def audio_playback_worker():

    stream = sd.OutputStream(
        samplerate=24000,
        channels=1,
        dtype="float32",
        blocksize=0,
        latency="low",
        #AIの修正案（低地円すぎるとぶちぶちする？）
        #blocksize=2048,
        #latency=0.05,
    )

    stream.start()

    print("\n[AUDIO WORKER STARTED]\n")

    try:

        while True:

            chunk = await audio_queue.get()

            if chunk is None:
                break

            #発話中フラグ
            IS_AI_SPERKING = True

            if interrupt_event.is_set():
                continue

            chunk = np.asarray(
                chunk,
                dtype=np.float32
            )

            chunk = chunk.reshape(-1, 1)

            stream.write(chunk)

            #発話終了
            IS_AI_SPERKING = False

    except Exception as e:

        print("\n[PLAYBACK ERROR]")
        print(e)

    finally:

        stream.stop()
        stream.close()

# ============================================================
# PIPELINE
# ============================================================

async def realtime_pipeline(audio):

    global audio_task

    interrupt_event.clear()

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

    print("min:", audio_np.min())
    print("max:", audio_np.max())

    # ========================================================
    # silence check
    # ========================================================

    #マイク入力の大きさを基準に
    peak = np.max(np.abs(audio_np))

    threshold = 0.9

    if IS_AI_SPERKING:
        threshold = 9.0   # ★10倍にする

    print("peak:", peak)
    print("threshold:", threshold)


    if peak < threshold:

        print("\n[SKIP] silence\n")
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
            return

        print("\n[AI]")
        print(partial)

        await tts_text_queue.put(partial)

# ============================================================
# CALLBACK
# ============================================================

def response(audio):

    try:

        future = asyncio.run_coroutine_threadsafe(
            realtime_pipeline(audio),
            loop
        )

        future.result()

    except Exception as e:

        print("\n[PIPELINE ERROR]")
        print(e)

    silence = np.zeros(
        2400,
        dtype=np.float32
    )

    yield (24000, silence)

# ============================================================
# FASTRTC
# ============================================================

stream = Stream(
    ReplyOnPause(
        response,
        can_interrupt=True,
    ),
    modality="audio",
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

    loop.create_task(
        audio_playback_worker()
    )

    loop.run_forever()

threading.Thread(
    target=loop_runner,
    daemon=True,
).start()

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
