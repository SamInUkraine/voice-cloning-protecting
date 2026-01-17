import streamlit as st
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from speechbrain.inference import EncoderClassifier
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from sklearn.decomposition import PCA

st.set_page_config(page_title="Voice Protection Test Bench", layout="wide")

SR = 16000

def load_wav(path: str, sr: int = SR) -> torch.Tensor:
    try:
        wav, r = torchaudio.load(path)
    except RuntimeError as e:
        if "torchcodec" in str(e).lower():
            data, r = sf.read(path, always_2d=True)
            wav = torch.from_numpy(data.T).float()
        else:
            raise
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if r != sr:
        wav = torchaudio.functional.resample(wav, r, sr)
    wav = wav.clamp(-1.0, 1.0)
    return wav.squeeze(0)

def chunk(x: torch.Tensor, chunk_sec=2.0, hop_sec=1.0, sr: int = SR, max_segs=30):
    cs = int(chunk_sec * sr)
    hs = int(hop_sec * sr)
    out = []
    for start in range(0, max(1, x.numel() - cs + 1), hs):
        seg = x[start:start+cs]
        if seg.numel() == cs:
            out.append(seg)
        if len(out) >= max_segs:
            break
    return out

@torch.no_grad()
def embed(spk, x: torch.Tensor, device: str):
    e = spk.encode_batch(x.unsqueeze(0).to(device)).squeeze()
    e = e / (e.norm(p=2) + 1e-12)
    return e.cpu()

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sum(a*b).item())

def rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x**2) + 1e-12).item())

def snr_db(clean: torch.Tensor, adv: torch.Tensor) -> float:
    L = min(clean.numel(), adv.numel())
    c, a = clean[:L], adv[:L]
    n = a - c
    return 20.0 * np.log10((rms(c) + 1e-12) / (rms(n) + 1e-12))

def eer(scores_same, scores_diff):
    all_scores = sorted(set(scores_same + scores_diff))
    best = (1.0, None)
    for t in all_scores:
        FAR = sum(s >= t for s in scores_diff) / max(1, len(scores_diff))
        FRR = sum(s <  t for s in scores_same) / max(1, len(scores_same))
        if abs(FAR - FRR) < best[0]:
            best = (abs(FAR - FRR), (FAR + FRR) / 2)
    return best[1] if best[1] is not None else 1.0

@torch.no_grad()
def asr_wav2vec2(x: torch.Tensor, device: str):
    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    mdl = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    inp = proc(x.numpy(), sampling_rate=SR, return_tensors="pt", padding=True)
    logits = mdl(inp.input_values.to(device)).logits
    pred = torch.argmax(logits, dim=-1)
    return proc.batch_decode(pred)[0].strip()

st.title("Voice Protection Test Bench (Safe Evaluation)")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"Device: {device}")

colA, colB = st.columns(2)
with colA:
    clean_path = st.text_input("Path to clean wav", r"D:\Backup\Python project\voice-cloning-protecting\voice_clean.wav")
with colB:
    prot_path = st.text_input("Path to protected wav", r"D:\Backup\Python project\voice-cloning-protecting\chunk_prot.wav")

chunk_sec = st.slider("Chunk length (sec)", 1.0, 4.0, 2.0, 0.5)
hop_sec = st.slider("Hop (sec)", 0.25, 2.0, 1.0, 0.25)
max_segs = st.slider("Max segments", 6, 60, 12, 2)

if st.button("Run evaluation"):
    clean = load_wav(clean_path)
    prot = load_wav(prot_path)

    st.subheader("Listen")
    st.write("Clean")
    st.audio(clean.numpy(), sample_rate=SR)
    st.write("Protected")
    st.audio(prot.numpy(), sample_rate=SR)

    st.subheader("Perceptibility")
    d = (prot[:min(len(clean), len(prot))] - clean[:min(len(clean), len(prot))])
    st.write({
        "SNR(dB)": float(snr_db(clean, prot)),
        "max|delta|": float(d.abs().max().item()),
        "RMS(delta)": float(rms(d))
    })

    st.subheader("Speaker verification metrics (ECAPA)")
    spk = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    segs_c = chunk(clean, chunk_sec, hop_sec, SR, max_segs=max_segs)
    segs_p = chunk(prot,  chunk_sec, hop_sec, SR, max_segs=max_segs)
    n = min(len(segs_c), len(segs_p))
    segs_c, segs_p = segs_c[:n], segs_p[:n]

    embs_c = [embed(spk, s, device) for s in segs_c]
    embs_p = [embed(spk, s, device) for s in segs_p]

    same = []
    for i in range(n):
        for j in range(i+1, n):
            same.append(cosine(embs_c[i], embs_c[j]))

    diff = [cosine(embs_c[i], embs_p[i]) for i in range(n)]

    st.write({
        "Mean cosine same(orig_i, orig_j)": float(np.mean(same)),
        "Mean cosine diff(orig_i, prot_i)": float(np.mean(diff)),
        "EER (same vs orig-protected)": float(eer(same, diff))
    })

    fig = plt.figure()
    plt.plot(diff, marker="o")
    plt.title("Per-segment cosine(orig_i, prot_i)")
    plt.xlabel("segment i")
    plt.ylabel("cosine similarity")
    st.pyplot(fig)

    # PCA візуалізація 
    X = torch.stack(embs_c + embs_p).numpy()
    labels = (["clean"]*n) + (["protected"]*n)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    fig2 = plt.figure()
    for lab in ["clean", "protected"]:
        idx = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(Z[idx,0], Z[idx,1], label=lab)
    plt.title("Embedding PCA (clean vs protected)")
    plt.legend()
    st.pyplot(fig2)

    st.subheader("ASR sanity check (Wav2Vec2)")
    with st.spinner("Transcribing..."):
        txt_c = asr_wav2vec2(clean.cpu(), device)
        txt_p = asr_wav2vec2(prot.cpu(), device)
    st.write("Clean transcript:", txt_c)
    st.write("Protected transcript:", txt_p)
