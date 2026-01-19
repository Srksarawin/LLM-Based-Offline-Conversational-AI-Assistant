import streamlit as st
import torch
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
import pyttsx3
from faster_whisper import WhisperModel
from llama_cpp import Llama
import os
import multiprocessing
import time

# ----------------------------
# CONFIGURATION
# ----------------------------
LLAMA_MODEL_PATH = r"C:\Users\NEW\Desktop\AI Assistant\models\Llama-3.2-1B-Instruct-Q4_K_L.gguf"
WHISPER_MODEL_PATH = r"C:\Users\NEW\Desktop\AI Assistant\models\faster-whisper-small"

RECORD_SECONDS = 5       # default recording time
SAMPLERATE = 44100
CHANNELS = 1

# ----------------------------
# LOAD MODELS (CACHED)
# ----------------------------
@st.cache_resource
def load_models():
    # Faster-Whisper (CPU)
    whisper = WhisperModel(
        WHISPER_MODEL_PATH,
        device="cpu",
        compute_type="int8",
        local_files_only=True
    )

    # LLaMA (CPU-friendly)
    n_threads = max(1, multiprocessing.cpu_count() - 1)
    llm = Llama(
        model_path=LLAMA_MODEL_PATH,
        n_ctx=384,
        n_threads=n_threads,
        n_gpu_layers=0,
        verbose=False
    )

    return whisper, llm

whisper, llm = load_models()

# ----------------------------
# INITIALIZE pyttsx3 (offline TTS)
# ----------------------------
@st.cache_resource
def init_tts():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # default voice
    engine.setProperty('rate', 150)           # speaking speed
    return engine

tts_engine = init_tts()

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.title("AI Assistant")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# HELPER: GENERATE RESPONSE
# ----------------------------
def generate_response(user_input: str) -> str:
    # Include last 10 messages for context
    context = ""
    for role, msg in st.session_state.history[-10:]:
        prefix = "User:" if role == "user" else "AI:"
        context += f"{prefix} {msg}\n"

    prompt = f"{context}User: {user_input}\nAI (Please answer in a detailed and elaborative manner, with examples if applicable):"

    output = llm.create_completion(
        prompt=prompt,
        max_tokens=150,   # increase token limit for longer responses
        stop=["User:", "AI:"]
    )
    return output["choices"][0]["text"].strip()

# ----------------------------
# USER INPUT
# ----------------------------
input_mode = st.radio("Choose Input Mode:", ["Text", "Voice"])
user_input = ""

# ---- TEXT INPUT ----
if input_mode == "Text":
    user_input = st.text_input("Type your message:")
    if st.button("Send") and user_input.strip():
        st.session_state.history.append(("user", user_input))

# ---- VOICE INPUT ----
elif input_mode == "Voice":
    if st.button("🎤 Record Voice"):
        st.info(f"Recording {RECORD_SECONDS} seconds...")
        audio_data = sd.rec(
            int(RECORD_SECONDS * SAMPLERATE),
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            dtype="int16"
        )

        # simple progress bar
        progress_bar = st.progress(0)
        for i in range(RECORD_SECONDS):
            time.sleep(1)
            progress_bar.progress((i + 1) / RECORD_SECONDS)

        sd.wait()

        wav_path = os.path.join(tempfile.gettempdir(), "input.wav")
        write(wav_path, SAMPLERATE, audio_data)
        st.success("✅ Recording complete")
        st.info("Transcribing...")

        # Whisper transcription
        segments, _ = whisper.transcribe(wav_path, beam_size=5, language="en")
        transcript = " ".join(seg.text for seg in segments)

        st.text_area("🗨 You said:", transcript, height=80)
        user_input = transcript

        if user_input.strip():
            st.session_state.history.append(("user", user_input))

# ----------------------------
# AI RESPONSE
# ----------------------------
if user_input.strip():
    with st.spinner("🤖 Generating response..."):
        response = generate_response(user_input)
        st.session_state.history.append(("ai", response))

    # Display chat in chronological order
    st.divider()
    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 AI:** {msg}")

    # ----------------------------
    # TEXT TO SPEECH (offline)
    # ----------------------------
    st.info("🔊 Generating speech (offline)...")
    mp3_path = os.path.join(tempfile.gettempdir(), "response.mp3")
    tts_engine.save_to_file(response, mp3_path)
    tts_engine.runAndWait()

    with open(mp3_path, "rb") as f:
        st.audio(f.read(), format="audio/mp3")

    st.success("🎧 Done!")
