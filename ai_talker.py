"""
====================================================================
RTX3080向け 超低遅延リアルタイム音声AI
FastRTC + Qwen3-ASR + Qwen3 4B Q4 + Qwen3-TTS
Voice Clone選択/録音対応 完全版
====================================================================

追加機能
--------------------------------------------------------------------
✓ 起動時 Voice Clone選択
✓ Voice録音モード
✓ wav/mp3/flac対応
✓ Voice Profile保存
✓ normalize
✓ silence trim
✓ Streaming TTS
✓ Streaming LLM
✓ 割り込み対応
✓ Audio Playback Cancel
✓ WebRTC

====================================================================
pip install
====================================================================

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install numpy soundfile librosa sounddevice

pip install "fastrtc[vad]"

pip install transformers accelerate sentencepiece

pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

pip install flash-attn --no-build-isolation

pip install xformers

====================================================================
必要モデル
====================================================================

ASR:
Qwen/Qwen3-ASR-0.6B

LLM:
Qwen3-4B-Instruct-Q4_K_M.gguf

TTS:
Qwen/Qwen3-TTS-0.6B-Realtime

====================================================================
"""

import os
import asyncio
import time
from pathlib import Path

# CUDA fragmentation対策
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf

import torch

# FlashAttention
torch.backends.cuda.enable_flash_sdp(True)

from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
)

from transformers import (
    Qwen3TTSForConditionalGeneration
)

from llama_cpp import Llama

from fastrtc import (
    Stream,
    ReplyOnPause,
)

# ============================================================
# CONFIG
# ============================================================

ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"

LLM_GGUF = "./models/Qwen3-4B-Instruct-Q4_K_M.gguf"

TTS_MODEL = "Qwen/Qwen3-TTS-0.6B-Realtime"

VOICE_DIR = "./voices"

N_CTX = 4096

os.makedirs(VOICE_DIR, exist_ok=True)

# ============================================================
# GLOBAL
# ============================================================

interrupt_event = asyncio.Event()

conversation_history = [
    {
        "role": "system",
        "content": """
あなたはリアルタイム音声AIです。

- 短く自然に返答
- 人間らしく話す
- 会話テンポ優先
- 長文禁止
"""
    }
]

REFERENCE_AUDIO = None

audio_task = None

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
    # MIC MODE
    # ========================================================

    if mode == "2":

        REFERENCE_AUDIO = record_reference_voice()

        return

    # ========================================================
    # EXTERNAL FILE
    # ========================================================

    elif mode == "3":

        path = input(
            "\n音声ファイルパス: "
        ).strip()

        processed = preprocess_reference_audio(
            path
        )

        REFERENCE_AUDIO = processed

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

    processed = preprocess_reference_audio(
        selected
    )

    REFERENCE_AUDIO = processed

    print(f"\n選択: {REFERENCE_AUDIO}\n")

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
# SELECT VOICE
# ============================================================

select_voice()

# ============================================================
# LOAD ASR
# ============================================================

print("Loading Qwen3-ASR...")

asr_processor = AutoProcessor.from_pretrained(
    ASR_MODEL
)

asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    ASR_MODEL,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

print("ASR OK")

# ============================================================
# LOAD LLM
# ============================================================

print("Loading Qwen3 4B Q4...")

llm = Llama(
    model_path=LLM_GGUF,
    n_gpu_layers=-1,
    n_ctx=N_CTX,
    n_batch=512,
    use_mmap=True,
    verbose=False
)

print("LLM OK")

# ============================================================
# LOAD TTS
# ============================================================

print("Loading Qwen3-TTS...")

tts_processor = AutoProcessor.from_pretrained(
    TTS_MODEL
)

tts_model = Qwen3TTSForConditionalGeneration.from_pretrained(
    TTS_MODEL,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

print("TTS OK")

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

    inputs = asr_processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(asr_model.device)

    with torch.no_grad():

        ids = asr_model.generate(
            **inputs,
            max_new_tokens=128
        )

    text = asr_processor.batch_decode(
        ids,
        skip_special_tokens=True
    )[0]

    yield text

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
        max_tokens=256
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

    inputs = tts_processor(
        text=text,
        reference_audio=REFERENCE_AUDIO,
        return_tensors="pt"
    ).to(tts_model.device)

    streamer = tts_model.generate_stream(
        **inputs
    )

    async for audio_chunk in streamer:

        if interrupt_event.is_set():
            return

        chunk = audio_chunk.cpu().numpy()

        yield chunk.astype(np.float32)

# ============================================================
# AUDIO PLAYBACK
# ============================================================

async def play_audio(audio_stream):

    stream = sd.OutputStream(
        samplerate=24000,
        channels=1,
        dtype="float32",
        blocksize=2048
    )

    stream.start()

    try:

        async for chunk in audio_stream:

            if interrupt_event.is_set():
                break

            stream.write(chunk)

    finally:

        stream.stop()
        stream.close()

# ============================================================
# MAIN PIPELINE
# ============================================================

async def realtime_pipeline(audio):

    global audio_task

    interrupt_event.clear()

    sample_rate, audio_np = audio

    print("\n================================================")
    print("USER")
    print("================================================")

    audio_np = audio_np.astype(np.float32)

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
        return

    # ========================================================
    # LLM
    # ========================================================

    token_stream = stream_llm(
        final_text
    )

    # ========================================================
    # TOKEN CHUNK
    # ========================================================

    async for partial in token_chunker(
        token_stream
    ):

        if interrupt_event.is_set():
            return

        print("\n[AI]")
        print(partial)

        # ====================================================
        # TTS
        # ====================================================

        audio_stream = streaming_tts(
            partial
        )

        # ====================================================
        # playback cancel
        # ====================================================

        if audio_task:

            audio_task.cancel()

        audio_task = asyncio.create_task(
            play_audio(audio_stream)
        )

# ============================================================
# INTERRUPT
# ============================================================

def on_interrupt():

    print("\n==============================")
    print("INTERRUPT")
    print("==============================")

    interrupt_event.set()

# ============================================================
# FASTRTC CALLBACK
# ============================================================

def response(audio):

    asyncio.run(
        realtime_pipeline(audio)
    )

# ============================================================
# FASTRTC
# ============================================================

stream = Stream(
    ReplyOnPause(
        response,
        can_interrupt=True,
        on_interrupt=on_interrupt
    ),
    modality="audio",
    mode="send-receive"
)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("""
====================================================================
Realtime Voice AI
====================================================================

Features:
✓ Voice Clone Select
✓ Mic Voice Record
✓ Qwen3-ASR
✓ Qwen3 4B Q4
✓ Qwen3-TTS
✓ Streaming
✓ Interruptions
✓ WebRTC

Browser:
http://127.0.0.1:7860

====================================================================
""")

    stream.ui.launch(
        server_name="0.0.0.0",
        server_port=7860
    )