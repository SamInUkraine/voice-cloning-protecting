import argparse
import os

# Disable TorchCodec to avoid FFmpeg/DLL issues and let torchaudio fall back to soundfile/sox.
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

import torch
import torchaudio
import soundfile as sf
from torch.nn import functional as F

if not hasattr(torchaudio, "list_audio_backends"):
    def _dummy_list_audio_backends():
        return []
    torchaudio.list_audio_backends = _dummy_list_audio_backends




from speechbrain.inference.speaker import EncoderClassifier


def run_attack(
    wav_path: str,
    out_wav_path: str,
    eps: float,
    alpha: float,
    num_iters: int,
    device: torch.device,
):
    print(f"Device: {device}")
    print(f"Config: eps={eps}, alpha={alpha}, iters={num_iters}")

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb"
    ).to(device)
    classifier.eval()

    data, fs = sf.read(wav_path, dtype="float32")   # [time, channels] or [time]
    signal = torch.from_numpy(data).transpose(0, 1) if data.ndim == 2 else torch.from_numpy(data).unsqueeze(0)
    print(f"Loaded wav: {wav_path}, shape={signal.shape}, fs={fs}")

    if signal.dim() == 2:
        signal = signal.mean(dim=0, keepdim=True)  # [1, time]

    signal = signal.to(device)

    with torch.no_grad():
        max_val = signal.abs().max()
        if max_val > 1.0:
            signal = signal / max_val
            print(f"Signal normalized by factor {max_val:.4f}")

    with torch.no_grad():
        emb_clean = classifier.encode_batch(signal)
        emb_clean = emb_clean.view(-1)
        emb_clean = F.normalize(emb_clean, dim=0)

    print("Clean embedding computed.")

    # 4. Random start для δ
    delta = torch.empty_like(signal).uniform_(-eps, eps).to(device)
    delta.requires_grad_(True)
    print(f"Initial delta max abs: {delta.detach().abs().max().item():.6f}")

    for i in range(num_iters):
        x_adv = torch.clamp(signal + delta, -1.0, 1.0)

        emb_adv = classifier.encode_batch(x_adv)
        emb_adv = emb_adv.view(-1)
        emb_adv = F.normalize(emb_adv, dim=0)

        cos_sim = F.cosine_similarity(
            emb_clean.unsqueeze(0),
            emb_adv.unsqueeze(0)
        ).mean()
        loss = cos_sim

        classifier.zero_grad()
        if delta.grad is not None:
            delta.grad.zero_()

        loss.backward()

        with torch.no_grad():
            delta -= alpha * delta.grad.sign()
            delta.clamp_(-eps, eps)

        delta.requires_grad_(True)

        if i % 10 == 0 or i == num_iters - 1:
            print(
                f"Iter {i:03d} | cos_sim={cos_sim.item():.4f} | "
                f"delta_max={delta.detach().abs().max().item():.6f}"
            )

    with torch.no_grad():
        x_adv = torch.clamp(signal + delta, -1.0, 1.0)

    x_adv_cpu = x_adv.detach().cpu()
    sf.write(out_wav_path, x_adv_cpu.permute(1, 0).numpy(), fs)
    print(f"Saved protected wav to {out_wav_path}")

    with torch.no_grad():
        emb_adv_final = classifier.encode_batch(x_adv)
        emb_adv_final = emb_adv_final.view(-1)
        emb_adv_final = F.normalize(emb_adv_final, dim=0)
        final_cos = F.cosine_similarity(
            emb_clean.unsqueeze(0),
            emb_adv_final.unsqueeze(0)
        ).item()

    print(f"Final cosine similarity: {final_cos:.4f}")
    return final_cos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, default="voice_clean.wav")
    parser.add_argument("--eps", type=float, default=0.002)
    parser.add_argument("--alpha", type=float, default=0.0004)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--suffix", type=str, default="eps0002")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_name = f"voice_protected_{args.suffix}.wav"

    run_attack(
        wav_path=args.wav,
        out_wav_path=out_name,
        eps=args.eps,
        alpha=args.alpha,
        num_iters=args.iters,
        device=device,
    )
