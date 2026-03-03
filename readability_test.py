import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer # Бібліотека для розрахунку WER

def check_readability(original_path, protected_path):
    # Завантаження моделі (наприклад, для англійської або багатомовної)
    model_name = "facebook/wav2vec2-base-960h" 
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def transcribe(audio_path):
        speech, sr = librosa.load(audio_path, sr=16000)
        input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]

    text_orig = transcribe(original_path)
    text_prot = transcribe(protected_path)

    # Розрахунок помилки
    error_rate = wer(text_orig, text_prot)
    
    return text_orig, text_prot, error_rate

# Приклад виклику:
orig, prot, error = check_readability("eng_monologue_test.wav", "voice_protected_eps0005.wav")
print(f"Original: {orig}")
print(f"Protected: {prot}")
print(f"WER: {error:.2%}")