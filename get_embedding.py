import os

os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

import torch
import torchaudio
import soundfile as sf

if not hasattr(torchaudio, "list_audio_backends"):
    def _dummy_list_audio_backends():
        return []
    torchaudio.list_audio_backends = _dummy_list_audio_backends

from speechbrain.inference.speaker import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"

)
# загружаємо аудіо
wav_path = "voice_clean.wav"
data, fs = sf.read(wav_path, dtype="float32")  # [time, channels] or [time]
signal = torch.from_numpy(data).transpose(0, 1) if data.ndim == 2 else torch.from_numpy(data).unsqueeze(0)

print(f"Loaded wav: {wav_path}, shape={signal.shape}, fs={fs}")

if signal.dim() == 2:
    signal = signal.mean(dim=0, keepdim=True)  # [1, time]

signal = signal  # [1, time]

# рахуємо embedding
classifier.eval()
with torch.no_grad():
    embeddings = classifier.encode_batch(signal)  # [batch, emb_dim]
    emb = embeddings.squeeze(0)  # [emb_dim]

print(f"Embedding shape: {emb.shape}")
print("First 10 dims:", emb[:10])

# зберігаємо embedding
torch.save(emb, "embedding_clean.pt")
print("Saved embedding to embedding_clean.pt")
