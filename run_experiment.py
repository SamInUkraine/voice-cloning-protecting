import os
import pandas as pd
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# ВАЖНО: Адаптируйте импорты под ваши функции из репозитория
# Например, если функция защиты в protect_voice_pgd.py называется protect_audio:
# from protect_voice_pgd import protect_audio
# from metrics_audio import calculate_snr

def run_pipeline(meta_csv, output_csv, epsilon):
    print("Инициализация модели ECAPA-TDNN...")
    # Загружаем модель для извлечения признаков голоса
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")
    
    df = pd.read_csv(meta_csv)
    results = []

    for index, row in df.iterrows():
        audio_path = row['file_path']
        
        if not os.path.exists(audio_path):
            print(f"Файл не найден: {audio_path}. Пропускаем...")
            continue
            
        print(f"[{index+1}/{len(df)}] Обработка: Диктор {row['speaker_id']} | {row['gender']} | {row['language']}")
        
        # 1. Загрузка чистого аудио
        signal, fs = torchaudio.load(audio_path)
        
        # Приведение к 16kHz (стандарт для ECAPA-TDNN)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)
            fs = 16000

        # 2. Эмбеддинг чистого аудио (оригинал)
        with torch.no_grad():
            emb_clean = classifier.encode_batch(signal)
        
        # 3. Применение PGD атаки
        # ЗАМЕНИТЕ эту строку на вызов вашей реальной функции генерации:
        # signal_protected = protect_audio(signal, classifier, epsilon)
        
        # Временная заглушка для проверки работы скрипта:
        signal_protected = signal + (torch.randn_like(signal) * epsilon)
        
        # 4. Эмбеддинг защищенного аудио
        with torch.no_grad():
            emb_protected = classifier.encode_batch(signal_protected)
        
        # 5. Вычисление метрик (Сходство)
        cos_sim = torch.nn.functional.cosine_similarity(emb_clean, emb_protected).item()
        
        # Замените на вызов вашей функции SNR:
        # snr_val = calculate_snr(signal, signal_protected)
        snr_val = 25.0 # Заглушка
        
        # Сохранение результата
        results.append({
            'file': os.path.basename(audio_path),
            'speaker_id': row['speaker_id'],
            'gender': row['gender'],
            'language': row['language'],
            'epsilon': epsilon,
            'cosine_similarity': cos_sim,
            'snr': snr_val
        })

    # Сохранение всех данных в CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nГотово! Результаты сохранены в файл: {output_csv}")

if __name__ == "__main__":
    # Указываем пути к файлам
    meta_file = "dataset_meta.csv"
    
    # ВСТАНОВЛЮЄМО ЦІЛЬОВЕ ЗНАЧЕННЯ EPSILON
    TARGET_EPSILON = 0.005
    
    out_file = f"experiment_results_eps_{TARGET_EPSILON}.csv"
    
    print(f"\n--- Запуск эксперимента с Epsilon = {TARGET_EPSILON} ---")
    run_pipeline(meta_file, out_file, epsilon=TARGET_EPSILON)