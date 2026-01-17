import argparse
import math
import os
import pathlib
import shutil


os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier


_orig_symlink = pathlib.Path.symlink_to


def _safe_symlink(self, target, target_is_directory=False):
    try:
        return _orig_symlink(self, target, target_is_directory=target_is_directory)
    except OSError:
        self.parent.mkdir(parents=True, exist_ok=True)
        return shutil.copy2(target, self)


pathlib.Path.symlink_to = _safe_symlink

def load_wav(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).clamp(-1.0, 1.0)

def chunk(x: torch.Tensor, sr: int, chunk_sec: float, hop_sec: float):
    L = x.numel()
    cs = int(chunk_sec * sr)
    hs = int(hop_sec * sr)
    out = []
    for start in range(0, max(1, L - cs + 1), hs):
        seg = x[start:start+cs]
        if seg.numel() == cs:
            out.append(seg)
    return out

@torch.no_grad()
def embed(spk, x: torch.Tensor, device: str):
    e = spk.encode_batch(x.unsqueeze(0).to(device)).squeeze()
    e = e / (e.norm(p=2) + 1e-12)
    return e

def cosine(a, b):
    return float(torch.sum(a*b).item())

def pgd_protect(spk, x, eps, alpha, steps, device):
    x = x.to(device)
    with torch.no_grad():
        e0 = embed(spk, x, device)

    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(steps):
        x_adv = (x + delta).clamp(-1.0, 1.0)
        e_adv = spk.encode_batch(x_adv.unsqueeze(0)).squeeze()
        e_adv = e_adv / (e_adv.norm(p=2) + 1e-12)
        loss = torch.sum(e0 * e_adv)  # мінімізуємо cosine
        loss.backward()
        with torch.no_grad():
            delta -= alpha * delta.grad.sign()
            delta.clamp_(-eps, eps)
        delta.grad.zero_()
    return (x + delta.detach()).clamp(-1.0, 1.0)

def eer(scores_same, scores_diff):
    
    all_scores = sorted(set(scores_same + scores_diff))
    best = (1.0, None)
    for t in all_scores:
        FAR = sum(s >= t for s in scores_diff) / max(1, len(scores_diff))  # false accept
        FRR = sum(s <  t for s in scores_same) / max(1, len(scores_same))  # false reject
        if abs(FAR - FRR) < best[0]:
            best = (abs(FAR - FRR), (FAR + FRR) / 2)
    return best[1] if best[1] is not None else 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--alpha", type=float, default=0.0002)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--chunk_sec", type=float, default=2.0)
    ap.add_argument("--hop_sec", type=float, default=1.0)
    ap.add_argument("--max_segs", type=int, default=12)
    ap.add_argument("--protected_wav", default=None, help="Path to precomputed protected wav (optional)")

    args = ap.parse_args()

    sr = 16000
    x = load_wav(args.wav, sr)
    segs = chunk(x, sr, args.chunk_sec, args.hop_sec)[:args.max_segs]
    print(f"Segments: {len(segs)}")

    prot_segs = None
    if args.protected_wav:
        xp = load_wav(args.protected_wav, sr)
        prot_segs = chunk(xp, sr, args.chunk_sec, args.hop_sec)[:args.max_segs]
        if len(prot_segs) != len(segs):
            n = min(len(segs), len(prot_segs))
            segs = segs[:n]
            prot_segs = prot_segs[:n]

    spk = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": args.device},
    ).to(args.device)

    embs = [embed(spk, s, args.device).cpu() for s in segs]
    if prot_segs is None:
        prots = [pgd_protect(spk, s, args.eps, args.alpha, args.steps, args.device).cpu() for s in segs]
    else:
        prots = [p.cpu() for p in prot_segs]

    embs_p = [embed(spk, p, args.device).cpu() for p in prots]


    same = []
    for i in range(len(embs)):
        for j in range(i+1, len(embs)):
            same.append(cosine(embs[i], embs[j]))


    diff = [cosine(embs[i], embs_p[i]) for i in range(len(embs))]

    print(f"Mean cosine same(orig_i, orig_j): {sum(same)/len(same):.4f}")
    print(f"Mean cosine diff(orig_i, prot_i): {sum(diff)/len(diff):.4f}")
    print(f"EER (same vs orig-protected):     {eer(same, diff):.4f}")

if __name__ == "__main__":
    main()
