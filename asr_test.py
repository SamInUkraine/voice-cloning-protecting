import os

os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

import torchaudio
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from jiwer import wer

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


def transcribe(path):
    wav, sr = sf.read(path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    audio = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)  # [1, T]
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
        sr = 16000
    input_values = processor(audio.squeeze(), sampling_rate=sr, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = logits.argmax(dim=-1)
    return processor.decode(predicted_ids[0])

clean = transcribe("voice_clean.wav")
prot = transcribe("voice_protected_eps0001.wav")

print("Clean transcription:\n", clean)
print("\nProtected transcription:\n", prot)

reference = "This is a sample of my clean voice where I'm trying to sound natural to show every detail in my voice so the AI model can copy and clone it into other recording"

print("\nWER clean:", wer(reference, clean))
print("WER protected:", wer(reference, prot))
