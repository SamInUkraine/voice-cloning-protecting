import os
# Вимикаємо TorchCodec до імпорту torchaudio
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

import torch
import torchaudio
import pandas as pd
# Оновлений імпорт для уникнення DeprecationWarning
from speechbrain.inference.speaker import EncoderClassifier
from protect_chunkwise import pgd_chunk, load_wav 
import torch.nn.functional as F

def get_metrics(clean_sig, adv_sig, model, device):
    """Обчислення метрик ефективності та прозорості"""
    with torch.no_grad():
        # Переконуємось, що сигнали мають потрібну розмірність [batch, time]
        if clean_sig.dim() == 1:
            clean_sig = clean_sig.unsqueeze(0)
        if adv_sig.dim() == 1:
            adv_sig = adv_sig.unsqueeze(0)
            
        # 1. Косинусна подібність (ефективність захисту)
        emb_clean = model.encode_batch(clean_sig.to(device)).squeeze()
        emb_adv = model.encode_batch(adv_sig.to(device)).squeeze()
        sim = F.cosine_similarity(emb_clean.unsqueeze(0), emb_adv.unsqueeze(0)).item()
        
        # 2. SNR (акустична непомітність)
        noise = adv_sig - clean_sig
        res_pwr = torch.mean(clean_sig**2)
        noise_pwr = torch.mean(noise**2) + 1e-12
        snr = 10 * torch.log10(res_pwr / noise_pwr).item()
        
        # 3. Максимальне збурення
        max_delta = torch.max(torch.abs(noise)).item()
        
    return sim, snr, max_delta

def run_test_suite(input_wav):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = 16000
    
    # Завантаження моделі через актуальний інтерфейс
    spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    
    # Завантажуємо оригінал
    if not os.path.exists(input_wav):
        print(f"Помилка: Файл {input_wav} не знайдено!")
        return

    clean = load_wav(input_wav, sr)
    
    # Параметри для тестування (різні рівні epsilon)
    eps_values = [0.0005, 0.001, 0.002, 0.005]
    results = []

    print(f"--- Тестування: {input_wav} (Device: {device}) ---")
    
    for eps in eps_values:
        # Генерація захищеного аудіо
        protected = pgd_chunk(spk, clean, eps=eps, alpha=eps/4, steps=20, 
                              device=device, lambda_mel=0.3)
        
        sim, snr, delta = get_metrics(clean, protected, spk, device)
        
        results.append({
            "Epsilon (ε)": eps,
            "SNR (dB)": round(snr, 2),
            "Cos Similarity": round(sim, 4),
            "Max Delta": round(delta, 5),
            "Status": "Захищено" if sim < 0.25 else "Вразливо"
        })

    # Вивід результатів
    df = pd.DataFrame(results)
    print("\nРЕЗУЛЬТАТИ ТЕСТУВАННЯ ЕФЕКТИВНОСТІ ЗАХИСТУ:")
    print(df.to_string(index=False))
    df.to_csv("test_results.csv", index=False)

if __name__ == "__main__":
    # Переконайтеся, що файл voice_clean.wav лежить у тій самій папці
    run_test_suite("voice_clean.wav")