"""
====================================================================
リアルタイム音声AI（MioTTS 直接組み込み版）
====================================================================

====================================================================
pip install
====================================================================

# CUDA 12.4
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# MioCodec（直接組み込み用）
pip install git+https://github.com/Aratako/MioCodec

# FastRTC
pip install "fastrtc[vad]"

# llama.cpp CUDA
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# その他
pip install numpy librosa soundfile sounddevice faster-whisper silero-vad gradio uvicorn fastapi emoji

====================================================================
必要モデル（ダウンロード先）
====================================================================

会話用 LLM:  ./models/Qwen3.5-2B-IQ4_XS.gguf
MioTTS:      ./models/MioTTS-0.6B-BF16.gguf   （または Q4_K_M）
MioCodec:    Aratako/MioCodec-25Hz-44.1kHz-v2（自動ダウンロード）
ASR:         faster-whisper large-v3-turbo（自動ダウンロード）

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
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# ffmpeg / sox（Windows の場合、パスを通しておく）
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
import gc
import threading
import re
import emoji
import inspect

from faster_whisper import WhisperModel
from llama_cpp import Llama
from fastrtc import Stream, StreamHandler
import silero_vad

# ---- MioTTS 用の追加インポート ----
from miocodec import MioCodecModel

# ---- Gradio + FastAPI ----
import gradio as gr
import uvicorn
from fastapi import FastAPI

# ============================================================
# CONFIG
# ============================================================

VOICE_DIR = "./voices"                         # 参照音声ファイル格納ディレクトリ
LLM_GGUF = "./models/Qwen3.5-2B-IQ4_XS.gguf"  # 会話用 LLM（GGUF）
MIOTTS_GGUF = "./models/MioTTS-0.6B-BF16.gguf"  # MioTTS 音声生成用 GGUF
CODEC_MODEL = "Aratako/MioCodec-25Hz-24kHz"  # MioCodec モデル（Hugging Face）
WHISPER_MODEL = "large-v3-turbo"

N_CTX_LLM = 4096      # 会話用 LLM のコンテキスト長
N_CTX_MIOTTS = 8192   # MioTTS 推奨コンテキスト長

# グローバル変数（音声クローン情報）
REFERENCE_TEXT = ""
VOICE_PROMPT = None   # MioTTS では global_embedding（torch.Tensor）を保持

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
    print("GPU:", torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram:.1f} GB")
print("================================================\n")

# ============================================================
# GLOBAL（非同期キュー・タスク管理）
# ============================================================

interrupt_event = asyncio.Event()
tts_text_queue = asyncio.Queue()
REFERENCE_AUDIO = None

current_pipeline_task = None
current_tts_task = None
generator_lock = threading.Lock()

# ============================================================
# 音声感度設定（VAD / 割り込み判定用）
# ============================================================

CHECK_INTERVAL = 0.5          # AI発話監視間隔（秒）
COOL_TIME = 0.7               # AI最終発話後のクールタイム（エコー対策）
MIC_THRESHOLD = 0.4           # マイク感度（通常時）
MIC_THRESHOLD_AI_SPERK = 0.6  # マイク感度（AI発話中）
CHECK_THRESHOLD = 0.015       # 割り込み判定値（通常時）
CHECK_THRESHOLD_AI_SPERK = 0.4 # 割り込み判定値（AI発話中）

# ============================================================
# RING BUFFER（再生用 PCM データ蓄積）
# ============================================================

RING_BUFFER_SIZE = 24000 * 20  # 20秒分（24kHz）
audio_ring = np.zeros(RING_BUFFER_SIZE, dtype=np.float32)
ring_write_pos = 0
ring_read_pos = 0
ring_lock = threading.Lock()

# ============================================================
# AUDIO BUFFER SYSTEM
# ============================================================

AUDIO_BUFFER_CHUNKS = 2
IS_AI_SPERKING = False
LAST_AI_SPEAK_TIME = 0.0

MAX_HISTORY_TURNS = 6  # 会話履歴保持数（ユーザー＋アシスタントで6往復）

# LLM 用システムプロンプト
conversation_history = [
    {
        "role": "system",
        "content": """
あなたは対話型AIです。ルールを厳守し、応答してください。

# ルール:
- あなたは音声読み上げ前提で応答する。箇条書き禁止。絵文字、記号禁止。読上可能な文章で応答する。
- キャラクター設定になりきって回答する。
- 過剰な長文、短文での応対は禁止。
- 会話の流れを常に意識し、過去の発言内容を踏まえた自然な応答を心がけること。
- ユーザーが話した内容を適切に引用・参照しながら返答すること。

# キャラクター設定:
- 優しい男性。娘に話しかけるように。
- 難しい言葉を避け、丁寧に応対する。
"""
    }
]

# ============================================================
# VOICE FILES（音声ファイル一覧・前処理）
# ============================================================

def list_voice_files():
    """VOICE_DIR 以下の音声ファイル一覧を返す"""
    exts = ["*.wav", "*.mp3", "*.flac"]
    files = []
    for ext in exts:
        files.extend(Path(VOICE_DIR).glob(ext))
    return files

def preprocess_reference_audio(path):
    """
    参照音声を 24kHz モノラルに変換し、トリム・正規化して保存する。
    戻り値: 処理後のファイルパス
    """
    path = path.strip().strip('"').strip("'")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print(f"\nLoading voice: {path}")
    audio, sr = librosa.load(path, sr=24000, mono=True)
    # 無音部分をトリム
    audio, _ = librosa.effects.trim(audio, top_db=20)
    # 正規化（ピークを 1.0 に）
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    # 保存
    save_path = os.path.join(VOICE_DIR, f"processed_{int(time.time())}.wav")
    sf.write(save_path, audio, 24000)
    return save_path

def record_reference_voice():
    """マイクで 20 秒録音し、参照音声として保存する"""
    print("\n================================================")
    print("Voice Clone録音")
    print("20秒間話してください")
    print("================================================\n")
    samplerate = 24000
    duration = 20
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    raw_path = os.path.join(VOICE_DIR, f"mic_voice_{int(time.time())}.wav")
    sf.write(raw_path, audio, samplerate)
    processed = preprocess_reference_audio(raw_path)
    print(f"\n保存: {processed}\n")
    return processed

# ============================================================
# LOAD ASR（faster-whisper）
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

def generate_reference_text(audio_path: str) -> str:
    """参照音声を ASR で文字起こしする"""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio = audio.astype(np.float32)
    segments, _ = asr_model.transcribe(audio, language="ja", beam_size=1, vad_filter=True)
    text = "".join([s.text for s in segments]).strip()
    print("\n================================================")
    print("[REFERENCE TRANSCRIPT]")
    print(text)
    print("================================================\n")
    return text

# ============================================================
# LOAD LLM（会話用）
# ============================================================

print("Loading Qwen GGUF（会話用）...")
if not os.path.exists(LLM_GGUF):
    raise FileNotFoundError(f"GGUF not found:\n{LLM_GGUF}")

llm = Llama(
    model_path=LLM_GGUF,
    n_gpu_layers=-1,
    n_ctx=N_CTX_LLM,
    n_batch=512,
    flash_attn=False,
    use_mmap=True,
    use_mlock=False,
    verbose=False,
    offload_kqv=True,
    logits_all=True,
)
print("LLM OK")

# ============================================================
# LOAD MioTTS（音声生成用 GGUF + MioCodec）
# ============================================================

print("Loading MioTTS GGUF...")
miotts_llm = Llama(
    model_path=MIOTTS_GGUF,
    n_gpu_layers=-1,
    n_ctx=N_CTX_MIOTTS,
    n_batch=512,
    flash_attn=False,
    use_mmap=True,
    use_mlock=False,
    verbose=False,
    offload_kqv=True,
    logits_all=True,
)
print("MioTTS LLM OK")

print("Loading MioCodec...")
codec = MioCodecModel.from_pretrained(CODEC_MODEL)
codec = codec.eval().to("cuda")
print("MioCodec OK")

# ============================================================
# MioTTS 用ユーティリティ関数
# ============================================================

def normalize_text(text: str) -> str:
    """MioTTS 用のテキスト正規化（全角→半角、半角カナ→全角カナなど）"""
    # 全角英数字→半角
    full_to_half = {
        chr(f): chr(h) for f, h in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)) + list(range(0xFF10, 0xFF1A)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)) + list(range(0x30, 0x3A))
        )
    }
    text = text.translate(str.maketrans(full_to_half))
    # 半角カタカナ→全角カタカナ
    hw_katakana = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
    fw_katakana = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
    text = text.translate(str.maketrans(hw_katakana, fw_katakana))
    return text

# MioTTS の音声トークンは <|s_数字|> 形式で出力される
TOKEN_PATTERN = re.compile(r"<\|s_(\d+)\|>")

def parse_speech_tokens(text: str) -> list[int]:
    """MioTTS の出力から音声トークンを抽出"""
    tokens = [int(v) for v in TOKEN_PATTERN.findall(text)]
    if not tokens:
        raise ValueError("No speech tokens found in LLM output.")
    return tokens

def extract_global_embedding(audio_path: str) -> torch.Tensor:
    """参照音声から MioCodec 用の global_embedding を抽出"""
    #audio, sr = librosa.load(audio_path, sr=44100, mono=True)
    audio, sr = librosa.load(audio_path, sr=24000, mono=True)  # 24kHzで読み込む
    waveform = torch.from_numpy(audio).float().unsqueeze(0).to("cuda")
    with torch.no_grad():
        ref_features = codec.encode(waveform, return_content=False, return_global=True)
    return ref_features.global_embedding  # shape: (1, embedding_dim)

def generate_audio_with_miotts(text: str, global_embedding: torch.Tensor) -> np.ndarray:
    """MioTTS で音声を生成（非ストリーミング）"""
    # 1. テキスト正規化
    normalized = normalize_text(text)

    # 2. MioTTS（LLM）で音声トークンを生成
    response = miotts_llm.create_chat_completion(
        messages=[{"role": "user", "content": normalized}],
        temperature=0.8,
        max_tokens=700,
    )
    llm_output = response["choices"][0]["message"]["content"]

    # 3. 音声トークンを解析
    tokens = parse_speech_tokens(llm_output)
    token_tensor = torch.tensor(tokens, dtype=torch.long, device="cuda")

    # 4. MioCodec でデコード（PCM波形に変換）
    with torch.no_grad():
        audio_tensor = codec.decode(
            global_embedding=global_embedding,
            content_token_indices=token_tensor,
        )
    audio_np = audio_tensor.squeeze(0).cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)

    return audio_np.astype(np.float32)  # 24kHzで出力される
    

# ---- ウォームアップ（ダミー音声で MioTTS を初期化） ----
def warmup_miotts():
    """ダミー音声を使って MioTTS + MioCodec をウォームアップ"""
    print("[WARMUP] MioTTS warmup start...")
    dummy_audio = np.zeros(16000, dtype=np.float32)
    dummy_path = os.path.join(VOICE_DIR, "dummy_warmup.wav")
    sf.write(dummy_path, dummy_audio, 16000)
    try:
        dummy_emb = extract_global_embedding(dummy_path)
        _ = generate_audio_with_miotts("こんにちは", dummy_emb)
        print("[WARMUP] MioTTS warmup OK")
    except Exception as e:
        print(f"[WARMUP] MioTTS warmup failed: {e}")
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

# ============================================================
# 初期設定（UIで設定されるまでデフォルトの音声を使う）
# ============================================================

default_voices = list_voice_files()
print("[DEBUG] Voice files at startup:", default_voices)

if default_voices:
    # 最初の音声を参照音声として自動設定
    REFERENCE_AUDIO = str(default_voices[0])
    REFERENCE_TEXT = generate_reference_text(REFERENCE_AUDIO)
    VOICE_PROMPT = extract_global_embedding(REFERENCE_AUDIO)
    print(f"[INIT] Default voice: {os.path.basename(REFERENCE_AUDIO)}")
    # ウォームアップ（実音声で）
    warmup_miotts()
else:
    print("[WARN] No voice file found. Warmup with dummy voice...")
    warmup_miotts()
    print("[INIT] No voice clone prompt set. User must set from UI.")

# ============================================================
# STREAMING ASR（faster-whisper）
# ============================================================

async def streaming_asr(audio_np, sr):
    """音声データを受け取り、ストリーミング文字起こし（ジェネレータ）"""
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr != 16000:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

    segments, _ = asr_model.transcribe(
        audio_np,
        language="ja",
        vad_filter=True,
        beam_size=1,
    )
    text = "".join(s.text for s in segments)
    yield text.strip()

# ============================================================
# 会話履歴の制御
# ============================================================

def trim_conversation_history():
    """会話履歴を MAX_HISTORY_TURNS に基づいてトリム"""
    global conversation_history
    system_msg = conversation_history[0] if conversation_history and conversation_history[0]["role"] == "system" else None
    messages = conversation_history if system_msg is None else conversation_history[1:]
    max_messages = MAX_HISTORY_TURNS * 2
    if len(messages) > max_messages:
        messages = messages[-max_messages:]
        conversation_history = ([system_msg] if system_msg else []) + messages
        print(f"[TRIM] History trimmed to {len(conversation_history)} messages")

# ============================================================
# STREAMING LLM（Qwen GGUF）
# ============================================================

async def stream_llm(user_text):
    """ユーザーテキストを LLM に送り、ストリーミング応答を生成"""
    global conversation_history

    trim_conversation_history()

    conversation_history.append({"role": "user", "content": user_text})
    print("\n[HISTORY]:", conversation_history)

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
            token = chunk["choices"][0]["delta"].get("content", "")
            if not token:
                continue

            # Qwen の思考タグを除去
            if "<think>" in token:
                thinking = True
                continue
            if "</think>" in token:
                thinking = False
                continue
            if thinking:
                continue
            if "<|" in token:
                continue

            full_text += token
            yield token

        except Exception as e:
            print(f"\n[AI LLM ERROR] {type(e)}: {e}")
            continue

    conversation_history.append({"role": "assistant", "content": full_text})

# ============================================================
# TOKEN CHUNKER（文節単位で分割）
# ============================================================

async def token_chunker(token_stream):
    """LLM のトークンストリームを文末（。！？.!?\n）で区切って出力"""
    current = ""
    async for token in token_stream:
        current += token
        if any(current.endswith(p) for p in "。！？.!?\n"):
            yield current.strip()
            current = ""
    if current.strip():
        yield current.strip()

# ============================================================
# 絵文字除去
# ============================================================

def remove_emojis(text: str) -> str:
    """テキストから絵文字を除去（Gradio/LLM の安定動作のため）"""
    return emoji.replace_emoji(text, replace='')

# ============================================================
# STREAMING TTS（MioTTS 版）
# ============================================================

async def _streaming_tts_impl(text):
    """
    MioTTS で音声を生成し、リングバッファに書き込む。
    MioTTS は非ストリーミングなので、一括生成後にチャンク分割する。
    """
    global VOICE_PROMPT

    try:
        text = remove_emojis(text).strip()
        if not text:
            return

        print("\n[TTS START] MioTTS")
        print(text)
        t0 = time.perf_counter()

        if VOICE_PROMPT is None:
            print("[TTS ERROR] No voice embedding set.")
            return

        # MioTTS で音声生成（非ストリーミング）
        audio_24k = await asyncio.to_thread(
            generate_audio_with_miotts,
            text,
            VOICE_PROMPT,
        )

        print(f"[TTS GEN TIME] {time.perf_counter() - t0:.2f}s")

        # チャンク分割してリングバッファに書き込み（20ms = 480 samples @ 24kHz）
        chunk_size = 480  # 20ms
        for i in range(0, len(audio_24k), chunk_size):
            chunk = audio_24k[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            ring_write(chunk.astype(np.float32))
            await asyncio.sleep(0)

    except Exception as e:
        print(f"\n[TTS ERROR] {type(e)}: {e}")

# ============================================================
# TTS WORKER（キューからテキストを取り出して合成）
# ============================================================

async def tts_worker():
    """tts_text_queue からテキストを取り出し、_streaming_tts_impl を実行"""
    print("\n[TTS WORKER STARTED]\n")
    while True:
        text = await tts_text_queue.get()
        if text is None:
            break
        try:
            await _streaming_tts_impl(text)
        except Exception as e:
            print(f"\n[TTS WORKER ERROR] {type(e)}: {e}")

# ============================================================
# PIPELINE
# ============================================================

async def _realtime_pipeline_impl(audio):
    """ASR → LLM → TTS のパイプライン本体"""
    sample_rate, audio_np = audio

    print("\n================================================")
    print("USER")
    print("================================================")
    print(f"[AUDIO INFO] sr={sample_rate}, shape={audio_np.shape}, dtype={audio_np.dtype}")

    # ASR
    final_text = ""
    async for partial_text in streaming_asr(audio_np, sample_rate):
        final_text = partial_text
        print("\n[ASR]", partial_text)

    if not final_text.strip():
        print("\n[EMPTY ASR]\n")
        return

    # LLM（ストリーミング応答）
    print("\n[START LLM]\n")
    token_stream = stream_llm(final_text)
    async for partial in token_chunker(token_stream):
        print("\n[LLM]", partial)
        await tts_text_queue.put(partial)

# ============================================================
# RING BUFFER 操作関数
# ============================================================

def ring_available():
    """リングバッファの読み取り可能サイズを返す"""
    with ring_lock:
        return (ring_write_pos - ring_read_pos) % RING_BUFFER_SIZE

def ring_read(frames):
    """リングバッファから指定フレーム数読み出す"""
    global ring_read_pos
    out = np.zeros(frames, dtype=np.float32)

    with ring_lock:
        available = (ring_write_pos - ring_read_pos) % RING_BUFFER_SIZE
        if available == 0:
            return out

        read_frames = min(frames, available)
        end = ring_read_pos + read_frames

        if end < RING_BUFFER_SIZE:
            out[:read_frames] = audio_ring[ring_read_pos:end]
        else:
            first = RING_BUFFER_SIZE - ring_read_pos
            out[:first] = audio_ring[ring_read_pos:]
            out[first:read_frames] = audio_ring[:read_frames - first]

        ring_read_pos = (ring_read_pos + read_frames) % RING_BUFFER_SIZE

    return out

def ring_write(audio):
    """リングバッファに音声データを書き込む"""
    global ring_write_pos

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = len(audio)

    with ring_lock:
        end = ring_write_pos + n
        if end < RING_BUFFER_SIZE:
            audio_ring[ring_write_pos:end] = audio
        else:
            first = RING_BUFFER_SIZE - ring_write_pos
            audio_ring[ring_write_pos:] = audio[:first]
            audio_ring[:n - first] = audio[first:]

        ring_write_pos = (ring_write_pos + n) % RING_BUFFER_SIZE

# ============================================================
# FastRTC CALLBACK（VoiceAIHandler）
# ============================================================

class VoiceAIHandler(StreamHandler):
    """
    FastRTC 用音声ハンドラ。
    - receive(): マイク入力を受け取り、VAD で発話区間を検出
    - _process_audio(): 発話終了後にパイプラインを起動
    - emit(): リングバッファから音声を読み出して出力
    - _interrupt(): 新規発話時に前の音声を即座に停止
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

    # ---- 内部キューアクセス ----
    def _get_output_queue(self):
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

    # ---- 出力キュー監視（LAST_AI_SPEAK_TIME更新） ----
    def _check_output_queue(self):
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

    # ---- VAD（音声活動検出） ----
    def _is_speech(self, audio_chunk):
        global LAST_AI_SPEAK_TIME

        # 16kHz にリサンプル
        if self.sample_rate != 16000:
            import librosa
            chunk_16k = librosa.resample(audio_chunk, orig_sr=self.sample_rate, target_sr=16000)
        else:
            chunk_16k = audio_chunk

        # AI発話中の感度を下げる
        threshold = MIC_THRESHOLD
        since_ai = time.time() - LAST_AI_SPEAK_TIME
        if since_ai < COOL_TIME:
            threshold = MIC_THRESHOLD_AI_SPERK

        tensor = torch.from_numpy(chunk_16k).float()
        speech_prob = silero_vad.get_speech_timestamps(
            tensor,
            self.vad_model,
            sampling_rate=16000,
            threshold=threshold
        )
        return len(speech_prob) > 0

    # ---- 割り込み処理 ----
    def _interrupt(self):
        """新規発話時に前のAI音声を即座に停止する"""
        global current_pipeline_task, current_tts_task, ring_read_pos, ring_write_pos

        # タスクキャンセル
        for task in (current_pipeline_task, current_tts_task):
            if task and not task.done():
                task.cancel()

        # リングバッファクリア
        with ring_lock:
            ring_read_pos = 0
            ring_write_pos = 0
            audio_ring.fill(0)

        # FastRTC 出力キューをクリア
        self.clear_queue()

        # TTS テキストキューをクリア
        while not tts_text_queue.empty():
            try:
                tts_text_queue.get_nowait()
            except:
                break

    # ---- 音声チェック（共通関数化） ----
    def _check_audio(self, audio_np):
        """音量チェック・クールダウン・ピークチェック"""
        threshold = CHECK_THRESHOLD
        since_ai = time.time() - LAST_AI_SPEAK_TIME
        if since_ai < COOL_TIME:
            threshold = CHECK_THRESHOLD_AI_SPERK
        peak = np.max(np.abs(audio_np))
        return peak, threshold, peak >= threshold

    # ---- 入力フレーム受信 ----
    def receive(self, frame):
        sr, audio_np = frame

        # モノラル化・正規化
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0

        # バッファに蓄積
        with self.buffer_lock:
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_np))
            if len(self.audio_buffer) < self.chunk_samples:
                return

            chunk = self.audio_buffer[:self.chunk_samples]
            self.audio_buffer = self.audio_buffer[self.chunk_samples:]

            # VAD判定
            if self._is_speech(chunk):
                self.silence_counter = 0
                if self.speech_buffer is None:
                    self.speech_buffer = chunk
                else:
                    self.speech_buffer = np.concatenate((self.speech_buffer, chunk))
            else:
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

    # ---- 発話完了後のパイプライン起動 ----
    async def _process_audio(self, audio_np, original_sr):
        # モノラル化・正規化（念のため）
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)
        audio_np = audio_np.astype(np.float32)
        max_abs = np.abs(audio_np).max()
        if max_abs > 1.0:
            audio_np = audio_np / 32768.0

        # 音声チェック
        peak, threshold, is_valid = self._check_audio(audio_np)
        if not is_valid:
            print(f"\n[SKIP] silence. peak={peak}, threshold={threshold}")
            return

        # 割り込み実行
        print(f"\nSTART PIPELINE. peak={peak}, threshold={threshold}")
        self._interrupt()

        # タスク起動
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

    # ---- 出力フレーム要求 ----
    def emit(self):
        now = time.time()
        if now - self._last_check_time >= self._check_interval:
            self._last_check_time = now
            if self._is_check:
                self._check_output_queue()

        chunk_size = 960
        available = ring_available()
        if available < chunk_size:
            return None

        self._is_check = True
        chunk = ring_read(chunk_size)
        return (24000, chunk)

    # ---- ハンドラコピー ----
    def copy(self):
        return VoiceAIHandler()

# ============================================================
# GRADIO UI（音声クローン設定）
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
    VOICE_PROMPT = extract_global_embedding(processed)

    if os.path.exists(temp_path):
        os.remove(temp_path)

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
    VOICE_PROMPT = extract_global_embedding(processed)

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

    # 前処理済み音声としてそのまま設定（再前処理はしない）
    REFERENCE_AUDIO = voice_path
    REFERENCE_TEXT = generate_reference_text(voice_path)
    VOICE_PROMPT = extract_global_embedding(voice_path)
    return f"✅ 選択完了: {voice_name}"

# ============================================================
# FastRTC Stream 作成
# ============================================================

stream = Stream(VoiceAIHandler(), modality="audio")
print("[DEBUG] stream.ui:", stream.ui)

# ============================================================
# START PLAYBACK WORKER（バックグラウンドで TTS 処理）
# ============================================================

background_loop = asyncio.new_event_loop()

def loop_runner():
    asyncio.set_event_loop(background_loop)
    background_loop.create_task(tts_worker())
    background_loop.run_forever()

threading.Thread(target=loop_runner, daemon=True).start()

# ============================================================
# GRADIO UI（stream.ui にカスタムコンポーネントを追加）
# ============================================================

with stream.ui:
    with gr.Accordion("🎤 音声クローン設定（最初に設定）", open=False):
        with gr.Row():
            mic_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="マイクで録音（20秒）",
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

        # イベント接続
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

# ============================================================
# FastAPI アプリ作成 & マウント
# ============================================================

app = FastAPI()
stream.mount(app)  # WebRTCエンドポイント追加
app = gr.mount_gradio_app(app, stream.ui, path="/")  # Gradio UI を / にマウント

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("""
====================================================================
Realtime Voice AI（MioTTS 直接組み込み版）
====================================================================

Browser:
http://127.0.0.1:7860

====================================================================
""")

    uvicorn.run(app, host="0.0.0.0", port=7860)