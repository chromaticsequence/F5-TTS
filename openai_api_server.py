import io
import os
import soundfile as sf
import whisper # Re-add whisper
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Assuming F5TTS class is accessible via this import path
# Adjust if necessary based on your project structure
from f5_tts.api import F5TTS

# --- Configuration ---
# German model details from aihpi/F5-TTS-German
HF_REPO_ID = "aihpi/F5-TTS-German"
MODEL_CHECKPOINT = "F5TTS_Base/model_420000.safetensors" # Using highest step checkpoint
MODEL_VOCAB = "F5TTS_Base/vocab.txt" # Assuming vocab is in the same dir
MODEL_CONFIG_NAME = "F5TTS_Base" # Base model config used for finetuning

REFERENCE_AUDIO_PATH = "voice_wav/ElevenLabs_2025-05-04T17_00_01_Brian_pre_sp100_s50_sb75_se0_b_m2.wav" # User's German reference
# Determine device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# --- Global Variables ---
f5tts_model: F5TTS = None
whisper_model: whisper.Whisper = None # Re-add whisper model global
reference_text: str = None # Will be transcribed
target_sample_rate: int = None

# --- Pydantic Models ---
class TTSRequest(BaseModel):
    input: str = Field(..., description="The text to synthesize.")
    model: str = Field(default=MODEL_CONFIG_NAME, description="The TTS model to use (ignored, uses configured model).")
    voice: str = Field(default="german_ref", description="The voice to use (ignored, uses configured reference).")
    # Add other OpenAI parameters if needed (response_format, speed), but keep simple for now

# --- Helper Functions ---
def transcribe_reference(audio_path):
    """Transcribes the reference audio using Whisper."""
    global whisper_model
    print(f"Transcribing reference audio: {audio_path}...")
    try:
        # Load whisper model if not already loaded
        if whisper_model is None:
            print(f"Loading Whisper model (using device: {DEVICE})...")
            # Consider using a smaller model if VRAM is limited, e.g., "base", "small"
            whisper_model = whisper.load_model("medium", device=DEVICE)
            print("Whisper model loaded.")

        result = whisper_model.transcribe(audio_path, language="de") # Specify German language
        transcription = result["text"]
        print(f"Transcription successful: {transcription}")
        return transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe reference audio: {e}")

# --- FastAPI Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models and transcribe on startup
    global f5tts_model, reference_text, target_sample_rate, whisper_model
    print("Server starting up...")
    print(f"Using device: {DEVICE}")

    if not os.path.exists(REFERENCE_AUDIO_PATH):
         raise FileNotFoundError(f"Reference audio file not found: {REFERENCE_AUDIO_PATH}")

    # Transcribe reference audio first
    reference_text = transcribe_reference(REFERENCE_AUDIO_PATH)

    # Load F5TTS model using paths from the German repo
    print(f"Loading German F5TTS model: {HF_REPO_ID}/{MODEL_CHECKPOINT}...")
    try:
        # Construct full HF paths
        ckpt_file_hf = f"hf://{HF_REPO_ID}/{MODEL_CHECKPOINT}"
        vocab_file_hf = f"hf://{HF_REPO_ID}/{MODEL_VOCAB}"

        f5tts_model = F5TTS(
            model=MODEL_CONFIG_NAME, # Use the base config name
            ckpt_file=ckpt_file_hf,  # Path to the specific German checkpoint
            vocab_file=vocab_file_hf,# Path to the specific German vocab
            device=DEVICE
        )
        target_sample_rate = f5tts_model.target_sample_rate # Get sample rate from model
        print(f"German F5TTS model loaded. Target sample rate: {target_sample_rate} Hz.")
    except Exception as e:
        print(f"Error loading German F5TTS model: {e}")
        # Allow server to start but endpoint will fail
        f5tts_model = None
        reference_text = None # Clear ref text if model fails

    yield
    # Clean up resources on shutdown (optional)
    print("Server shutting down...")
    # Release GPU memory if needed
    del f5tts_model
    if whisper_model is not None: # Check if it was loaded before deleting
        del whisper_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# --- FastAPI App ---
app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---
@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    Generates audio from the input text using the configured F5TTS model
    and reference voice, mimicking the OpenAI TTS API structure.
    """
    global f5tts_model, reference_text, target_sample_rate

    if f5tts_model is None or reference_text is None:
        raise HTTPException(status_code=503, detail="TTS model or reference transcription not available.")

    print(f"Received request to synthesize: '{request.input}'")

    try:
        # Perform inference
        wav, sr, _ = f5tts_model.infer(
            ref_file=REFERENCE_AUDIO_PATH,
            ref_text=reference_text,
            gen_text=request.input,
            # nfe_step=16, # Removed to use default for potentially better quality
            # Use default inference parameters from F5TTS class or adjust as needed
            # show_info=print # Optional: uncomment for more verbose logging
        )

        # Ensure sample rate matches model's target
        if sr != target_sample_rate:
             print(f"Warning: Inference sample rate ({sr}) differs from target ({target_sample_rate}). This shouldn't happen.")
             # Potentially resample here if necessary, but F5TTS should return target_sample_rate

        # Convert numpy array to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, wav, target_sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        audio_bytes = buffer.read()

        print("Synthesis successful.")
        return Response(content=audio_bytes, media_type="audio/wav")

    except Exception as e:
        print(f"Error during TTS inference: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {e}")

@app.get("/")
async def root():
    return {"message": "F5-TTS OpenAI-Compatible API Server is running."}

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    # Use reload=True for development to automatically reload on code changes
    # Set workers > 1 for production if needed, but be mindful of VRAM per worker
    uvicorn.run("openai_api_server:app", host="0.0.0.0", port=8010, reload=False, workers=1)