import io
import os
import json # Add json import
import soundfile as sf
import whisper # Re-add whisper
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
# Removed cached_path import

# Assuming F5TTS class is accessible via this import path
# Adjust if necessary based on your project structure
from f5_tts.api import F5TTS

# --- Configuration ---
# German model details from aihpi/F5-TTS-German
HF_REPO_ID = "aihpi/F5-TTS-German"
MODEL_CHECKPOINT = "F5TTS_Base/model_295000.safetensors" # Trying the earliest checkpoint
MODEL_VOCAB = "vocab.txt" # Assuming vocab is at the repo root
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
whisper_model: whisper.Whisper = None
reference_text: str = None
processed_ref_audio_path: str = None # Store path to potentially clipped audio
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
    # Load models and preprocess/transcribe reference audio on startup
    global f5tts_model, reference_text, target_sample_rate, whisper_model, processed_ref_audio_path
    print("Server starting up...")
    print(f"Using device: {DEVICE}")

    if not os.path.exists(REFERENCE_AUDIO_PATH):
         raise FileNotFoundError(f"Reference audio file not found: {REFERENCE_AUDIO_PATH}")

    # --- Preprocess Reference Audio FIRST ---
    # This function clips the audio if needed and returns the path to the
    # potentially temporary clipped file, along with empty text if transcription is needed.
    # We need to import the function. Let's assume it's available.
    # We might need to adjust imports if this fails.
    try:
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text
        print(f"Preprocessing reference audio: {REFERENCE_AUDIO_PATH}...")
        # Pass empty string for ref_text to trigger transcription if needed later
        processed_ref_audio_path, ref_text_from_preprocess = preprocess_ref_audio_text(
            REFERENCE_AUDIO_PATH, "", clip_short=True, show_info=print
        )
        print(f"Reference audio processed. Using path: {processed_ref_audio_path}")
    except ImportError:
         raise RuntimeError("Could not import preprocess_ref_audio_text. Check installation.")
    except Exception as e:
        print(f"Error during reference audio preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to preprocess reference audio: {e}")

    # --- Transcribe the PROCESSED audio if needed ---
    if not ref_text_from_preprocess.strip():
        print("Transcription required for processed reference audio...")
        reference_text = transcribe_reference(processed_ref_audio_path) # Transcribe the clipped audio
    else:
        # This case shouldn't happen if we pass "" above, but handle defensively
        print("Using reference text determined during preprocessing (unexpected).")
        reference_text = ref_text_from_preprocess

    # --- Load F5TTS model ---
    print(f"Loading German F5TTS model: {HF_REPO_ID}/{MODEL_CHECKPOINT}...")
    try:
        # Define local paths using the cloned repo
        local_ckpt_path = os.path.join("german_model_repo", MODEL_CHECKPOINT)
        local_vocab_path = os.path.join("german_model_repo", MODEL_VOCAB)

        print(f"Using local checkpoint: {local_ckpt_path}")
        print(f"Using local vocab file: {local_vocab_path}")

        # Ensure local files exist before loading
        if not os.path.exists(local_ckpt_path):
            raise FileNotFoundError(f"Local checkpoint file not found: {local_ckpt_path}")
        if not os.path.exists(local_vocab_path):
             raise FileNotFoundError(f"Local vocab file not found: {local_vocab_path}")

        f5tts_model = F5TTS(
            model=MODEL_CONFIG_NAME,    # Use the base config name
            ckpt_file=local_ckpt_path,  # Path to the local German checkpoint
            vocab_file=local_vocab_path,# Path to the local German vocab
            device=DEVICE
        )
        target_sample_rate = f5tts_model.target_sample_rate # Get sample rate from model
        print(f"German F5TTS model loaded. Target sample rate: {target_sample_rate} Hz.")
        # Apply torch.compile optimization
        print("Applying torch.compile() optimization...")
        try:
            # Note: Compilation might take time on first inference after startup
            # Try specifying the inductor backend
            f5tts_model = torch.compile(f5tts_model, backend='inductor')
            print("torch.compile(backend='inductor') applied successfully.")
        except Exception as compile_e:
            print(f"Warning: torch.compile() failed: {compile_e}. Proceeding without optimization.")
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

    # Load parameters from JSON file for each request
    params_file = "inference_params.json"
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        print(f"Loaded inference parameters: {params}")
    except FileNotFoundError:
        print(f"Warning: '{params_file}' not found. Using default inference parameters.")
        params = {} # Use defaults if file not found
    except json.JSONDecodeError:
        print(f"Error: Could not decode '{params_file}'. Using default inference parameters.")
        params = {} # Use defaults if JSON is invalid

    try:
        # Perform inference using loaded parameters
        wav, sr, _ = f5tts_model.infer(
            ref_file=processed_ref_audio_path, # Use the processed (potentially clipped) audio path
            ref_text=reference_text,           # Use the corresponding transcription
            gen_text=request.input,
            **params # Pass parameters loaded from JSON
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