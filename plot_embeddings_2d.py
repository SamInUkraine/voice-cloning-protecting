import os

# Disable TorchCodec to avoid FFmpeg/DLL issues; we load audio via soundfile.
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

import torch
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf
from torch.nn.functional import normalize
from sklearn.decomposition import PCA

# PATCH torchaudio: stub missing list_audio_backends for SpeechBrain check.
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


files = [
    ("me_clean_1.wav",    "me_clean"),
    ("me_clean_2.wav",    "me_clean"),
    ("me_clean_3.wav",    "me_clean"),
    ("voice_protected_eps0001.wav", "me_protected"),
    ("voice_protected_eps0002.wav", "me_protected"),
    ("other_speaker_1.wav", "other"),
    ("other_speaker_2.wav", "other"),
]

embs = []
labels = []
missing = []
for path, label in files:
    if not os.path.exists(path):
        missing.append(path)
        print(f"Skipping missing file: {path}")
        continue
    print("Embedding:", path)
    emb = get_emb(path)
    embs.append(emb.numpy())
    labels.append(label)

if not embs:
    raise SystemExit("No input audio files were found; nothing to plot.")

import numpy as np
X = np.stack(embs, axis=0)   # [N, D]

pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

color_map = {
    "me_clean": "blue",
    "me_protected": "red",
    "other": "green",
}

plt.figure()
for (x, y), label, (path, cls) in zip(X2, labels, [f for f in files if os.path.exists(f[0])]):
    c = color_map[label]
    plt.scatter(x, y, c=c)
    plt.text(x + 0.02, y + 0.02, path, fontsize=8)

plt.title("Speaker embeddings projected to 2D (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

legend_elems = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='My clean voice'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',  label='My protected voice'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',label='Other speaker'),
]
plt.legend(handles=legend_elems)
plt.tight_layout()
plt.savefig("embeddings_pca.png", dpi=200)
print("Saved plot to embeddings_pca.png")
if missing:
    print("Missing files skipped:", ", ".join(missing))
