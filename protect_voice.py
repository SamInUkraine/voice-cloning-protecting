import argparse
import os
import torchaudio.transforms as T

# Avoid symlink creation on Windows when downloading models via huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import pathlib
import shutil

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from jiwer import wer


# SpeechBrain uses pathlib.Path.symlink_to when fetching HF assets; this fails on Windows without
# developer/admin privileges. Patch the method to fall back to copying when symlinks are not allowed.
_orig_symlink = pathlib.Path.symlink_to


def _safe_symlink(self, target, target_is_directory=False):
    try:
        return _orig_symlink(self, target, target_is_directory=target_is_directory)
    except OSError:
        self.parent.mkdir(parents=True, exist_ok=True)
        return shutil.copy2(target, self)


pathlib.Path.symlink_to = _safe_symlink

def load_wav(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # (ch, t)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.clamp(-1.0, 1.0)
    return wav.squeeze(0)  # (t,)

@torch.no_grad()
def spk_embed(model: EncoderClassifier, x: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    # SpeechBrain ожидает (batch, time)
    emb = model.encode_batch(x.unsqueeze(0))  # (1, 1, dim) или (1, dim) зависит от версии
    emb = emb.squeeze()
    emb = emb / (emb.norm(p=2) + 1e-12)
    return emb

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sum(a * b).item())

def pgd_protect(
    spk_model,
    x,
    eps=0.001,
    alpha=0.0002,
    steps=30,
    device="cpu",
    lambda_spk=1.0,
    lambda_mel=0.3,
):
    x = x.to(device)
    spk_model.to(device)

    mel = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    ).to(device)

    with torch.no_grad():
        e_orig = spk_model.encode_batch(x.unsqueeze(0)).squeeze()
        e_orig = e_orig / (e_orig.norm() + 1e-12)
        mel_orig = mel(x)

    delta = torch.zeros_like(x, requires_grad=True)

    for _ in range(steps):
        x_adv = (x + delta).clamp(-1.0, 1.0)

        e_adv = spk_model.encode_batch(x_adv.unsqueeze(0)).squeeze()
        e_adv = e_adv / (e_adv.norm() + 1e-12)

        mel_adv = mel(x_adv)

        loss_spk = torch.sum(e_orig * e_adv)          # хотим уменьшить cosine
        loss_mel = torch.mean((mel_orig - mel_adv) ** 2)  # хотим сохранить спектр

        loss = lambda_spk * loss_spk + lambda_mel * loss_mel
        loss.backward()

        with torch.no_grad():
            delta -= alpha * delta.grad.sign()
            delta.clamp_(-eps, eps)

        delta.grad.zero_()

    return (x + delta.detach()).clamp(-1.0, 1.0)


@torch.no_grad()
def asr_transcribe_wav2vec2(
    wav2vec2_processor: Wav2Vec2Processor,
    wav2vec2_model: Wav2Vec2ForCTC,
    x: torch.Tensor,
    device: str = "cpu",
    sr: int = 16000,
) -> str:
    wav2vec2_model.to(device)
    inputs = wav2vec2_processor(x.cpu().numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    logits = wav2vec2_model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = wav2vec2_processor.batch_decode(pred_ids)[0]
    return text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_wav", required=True)
    ap.add_argument("--out_wav", default="protected.wav")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eps", type=float, default=0.002)      # ~0.2% амплитуды
    ap.add_argument("--alpha", type=float, default=0.0004)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ref_text", default=None, help="Если есть эталонный текст — посчитаем WER")
    ap.add_argument("--lambda_mel", type=float, default=0.3)

    args = ap.parse_args()

    x = load_wav(args.in_wav, 16000)

    spk = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": args.device},
    )

    e0 = spk_embed(spk, x.to(args.device))
    sim_self = cosine(e0, e0)

    x_adv = pgd_protect(
        spk, x,
        eps=args.eps,
        alpha=args.alpha,
        steps=args.steps,
        device=args.device,
        lambda_mel=args.lambda_mel
    )

    e_adv = spk_embed(spk, x_adv.to(args.device))
    sim = cosine(e0, e_adv)

    out_dir = os.path.dirname(os.path.abspath(args.out_wav))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    torchaudio.save(args.out_wav, x_adv.unsqueeze(0).cpu(), 16000)

    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    clean_text = asr_transcribe_wav2vec2(proc, asr_model, x, device=args.device)
    adv_text = asr_transcribe_wav2vec2(proc, asr_model, x_adv, device=args.device)

    print("=== Speaker similarity ===")
    print(f"Self-check cosine(orig, orig): {sim_self:.4f}")
    print(f"Cosine(orig, protected):       {sim:.4f}")
    print()
    print("=== ASR transcripts (Wav2Vec2) ===")
    print("Clean:", clean_text)
    print("Protected:", adv_text)

    if args.ref_text:
        w_clean = wer(args.ref_text, clean_text)
        w_adv = wer(args.ref_text, adv_text)
        print()
        print("=== WER vs ref_text ===")
        print(f"WER clean:     {w_clean:.4f}")
        print(f"WER protected: {w_adv:.4f}")

    print()
    print(f"Saved protected wav -> {args.out_wav}")

if __name__ == "__main__":
    main()
