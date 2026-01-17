import os

os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

import torch
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf
from torch.nn.functional import normalize, cosine_similarity

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []

from speechbrain.inference.speaker import EncoderClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
).to(device)
classifier.eval()


def get_emb(path: str) -> torch.Tensor:
    wav, fs = sf.read(path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = torch.tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        emb = classifier.encode_batch(wav)
        emb = emb.view(-1)
        emb = normalize(emb, dim=0)
    return emb.cpu()


clean_path = "voice_clean.wav"
variants = {
    "eps = 0.0005": "voice_protected_eps00005.wav",
    "eps = 0.0010": "voice_protected_eps0001.wav",
    "eps = 0.0020": "voice_protected_eps0002.wav",
}

emb_clean = get_emb(clean_path)

labels = []
values = []

for label, path in variants.items():
    if not os.path.exists(path):
        print(f"Skipping {path} (file not found)")
        continue
    emb_prot = get_emb(path)
    cos = cosine_similarity(emb_clean, emb_prot, dim=0).item()
    labels.append(label)
    values.append(cos)
    print(label, "-> cosine similarity:", cos)

if not values:
    raise SystemExit("No variant files found; nothing to plot.")

plt.figure()
plt.bar(labels, values)
plt.ylim(-1.0, 1.0)
plt.axhline(0.0, linestyle="--")
plt.ylabel("Cosine similarity with clean embedding")
plt.title("Effect of L-infinity adversarial protection on speaker embedding")
plt.tight_layout()
plt.savefig("cosine_vs_eps.png", dpi=200)
print("Saved plot to cosine_vs_eps.png")
