import os
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

import argparse
import math
import torch
import torchaudio
import soundfile as sf  # Використовуємо soundfile напряму
import torchaudio.transforms as T
from speechbrain.inference.speaker import EncoderClassifier

def load_wav(path: str, target_sr: int = 16000) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(data)
    
    # Конвертація в моно
    if wav.ndim > 1:
        wav = wav.mean(dim=-1, keepdim=True)
    else:
        wav = wav.unsqueeze(-1)
    
    wav = wav.transpose(0, 1) # [channels, time]
    
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    
    return wav.squeeze(0).clamp(-1.0, 1.0)


def save_wav(path: str, x: torch.Tensor, sr: int = 16000):
    torchaudio.save(path, x.unsqueeze(0).cpu(), sr)

def hann(L: int, device: str):
    return torch.hann_window(L, periodic=False, device=device)

def pgd_chunk(spk, x, eps, alpha, steps, device, lambda_mel):
    x = x.to(device)

    mel = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80).to(device)

    with torch.no_grad():
        e0 = spk.encode_batch(x.unsqueeze(0)).squeeze()
        e0 = e0 / (e0.norm() + 1e-12)
        m0 = mel(x)

    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(steps):
        xa = (x + delta).clamp(-1.0, 1.0)

        ea = spk.encode_batch(xa.unsqueeze(0)).squeeze()
        ea = ea / (ea.norm() + 1e-12)
        ma = mel(xa)

        loss_spk = torch.sum(e0 * ea)             
        loss_mel = torch.mean((m0 - ma) ** 2)
        loss = loss_spk + lambda_mel * loss_mel

        loss.backward()
        with torch.no_grad():
            delta -= alpha * delta.grad.sign()
            delta.clamp_(-eps, eps)
        delta.grad.zero_()

    return (x + delta.detach()).clamp(-1.0, 1.0)

def protect_chunkwise(x, spk, sr, chunk_sec, hop_sec, eps, alpha, steps, device, lambda_mel):
    L = x.numel()
    cs = int(chunk_sec * sr)
    hs = int(hop_sec * sr)
    w = hann(cs, device)

    out = torch.zeros(L, device=device)
    wsum = torch.zeros(L, device=device)

    spk.to(device)

    for start in range(0, max(1, L - cs + 1), hs):
        chunk = x[start:start+cs]
        if chunk.numel() != cs:
            break
        adv = pgd_chunk(spk, chunk, eps, alpha, steps, device, lambda_mel)

        out[start:start+cs] += adv * w
        wsum[start:start+cs] += w

    out = out / (wsum + 1e-8)
    return out.clamp(-1.0, 1.0).cpu()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_wav", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--alpha", type=float, default=0.0002)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--chunk_sec", type=float, default=2.0)
    ap.add_argument("--hop_sec", type=float, default=1.0)
    ap.add_argument("--lambda_mel", type=float, default=0.3)
    args = ap.parse_args()

    sr = 16000
    x = load_wav(args.in_wav, sr)

    spk = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": args.device},
    )

    y = protect_chunkwise(
        x, spk, sr,
        chunk_sec=args.chunk_sec,
        hop_sec=args.hop_sec,
        eps=args.eps,
        alpha=args.alpha,
        steps=args.steps,
        device=args.device,
        lambda_mel=args.lambda_mel
    )

    save_wav(args.out_wav, y, sr)
    print("Saved:", args.out_wav)

if __name__ == "__main__":
    main()
