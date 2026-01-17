import os

# Force torchaudio to avoid torchcodec/FFmpeg backend on Windows
os.environ.setdefault("TORCHAUDIO_USE_LIBTORCHCODEC", "0")

import torch
import torch.nn.functional as F
import soundfile as sf
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
import TTS.tts.models.xtts as xtts_module
import TTS.utils.io as tts_io
from TTS.api import TTS

#тестовий скрипт для перевірки чи працює клонування голосу після захисту

add_safe_globals([XttsConfig, xtts_module.XttsAudioConfig, xtts_module.XttsArgs])


def _load_fsspec_allow_pickle(path, map_location=None, cache=True, **kwargs):
    kwargs.setdefault("weights_only", False)
    return tts_io.load_fsspec(path, map_location=map_location, cache=cache, **kwargs)


xtts_module.load_fsspec = _load_fsspec_allow_pickle


# Replace torchaudio.load with soundfile-based loader to avoid torchcodec issues
def _load_audio_without_torchcodec(audiopath, sampling_rate):
    audio_np, sr = sf.read(audiopath, dtype="float32")
    audio = torch.from_numpy(audio_np)

    # stereo to mono if needed
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.T
        audio = torch.mean(audio, dim=0, keepdim=True)

    if sr != sampling_rate:
        # Simple linear resample to target length
        target_len = int(audio.shape[-1] * sampling_rate / sr)
        audio = F.interpolate(
            audio.unsqueeze(0),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

    audio = audio.clamp(-1, 1)
    return audio


xtts_module.load_audio = _load_audio_without_torchcodec

# Choose device (XTTS runs faster on GPU, but CPU also works)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# Load model
print("Loading XTTS model... (this might take a minute)")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Text to synthesize
target_text = (
    "This is a demonstration of adversarial defense. "
    "If you hear my real voice, the defense failed. "
    "If I sound like a stranger, the defense worked."
)

# Reference audio
clean_ref = "voice_clean.wav"      # original voice
protected_ref = "p_0030.wav"       # protected/perturbed voice

# Clone original voice
print("\nAttempting to clone ORIGINAL voice...")
tts.tts_to_file(
    text=target_text,
    speaker_wav=clean_ref,
    language="en",
    file_path="output_clone_from_original.wav",
)
print("Done! Saved to 'output_clone_from_original.wav'")

# Clone protected voice (may fail or sound different if defense works)
print("\nAttempting to clone PROTECTED voice...")
try:
    tts.tts_to_file(
        text=target_text,
        speaker_wav=protected_ref,
        language="en",
        file_path="output_clone_from_protected.wav",
    )
    print("Done! Saved to 'output_clone_from_protected.wav'")
except Exception as e:
    print(f"Model crashed on protected audio (SUCCESS for defense): {e}")

print("\nTEST COMPLETE.")
print("Listen to:")
print("1) output_clone_from_original.wav -> should sound like you.")
print("2) output_clone_from_protected.wav -> should sound like a stranger or fail.")
