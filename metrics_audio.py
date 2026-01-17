import argparse
import math
import torch
import torchaudio

def load(path, sr=16000):
    x, r = torchaudio.load(path)
    if x.size(0) > 1:
        x = x.mean(dim=0, keepdim=True)
    if r != sr:
        x = torchaudio.functional.resample(x, r, sr)
    return x.squeeze(0)

def rms(x):
    return float(torch.sqrt(torch.mean(x**2) + 1e-12).item())

def snr_db(clean, adv):
    noise = adv - clean
    return 20.0 * math.log10((rms(clean) + 1e-12) / (rms(noise) + 1e-12))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--adv", required=True, nargs="+")
    args = ap.parse_args()

    clean = load(args.clean)

    print("file\tSNR(dB)\tmax|delta|\tRMS(delta)")
    for p in args.adv:
        adv = load(p)
        L = min(clean.numel(), adv.numel())
        c = clean[:L]
        a = adv[:L]
        d = a - c
        print(f"{p}\t{snr_db(c,a):.2f}\t{float(d.abs().max().item()):.6f}\t{rms(d):.6f}")

if __name__ == "__main__":
    main()
