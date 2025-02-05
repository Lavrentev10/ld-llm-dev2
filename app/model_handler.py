import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelHandler:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-7B"
        self.model_path = "/app/models/Qwen2.5-7B"

        print(os.path.join(self.model_path, "config.json"))
        print(os.path.exists(os.path.join(self.model_path, "config.json")))

        if not os.path.exists(os.path.join(self.model_path, "config.json")): # Проверить верность пусти для модели
            print("config.json не найден")
            # time.sleep(5)
            self.download_model()

        print(f"Загрузка: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        print("tokenizer загружен")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.float16
        )
        print(f"{self.model_path} загружена")

    def download_model(self):
        os.makedirs(self.model_path, exist_ok=True) # Создается не там, где надо, наверное, и проверка происходит тоже.
        print(f"Скачивание: {self.model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, force_download=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, force_download=True)

        tokenizer.save_pretrained(self.model_path)
        model.save_pretrained(self.model_path)

        config_path = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_path):
            print(f"Модель {self.model_name} сохранена в {self.model_path}")
        else:
            print("Ошибка: config.json не найдет")

    def get_recommendation(self, diagnosis, complication=None):
        prompt = f"Диагноз: {diagnosis}"
        if complication:
            prompt += f", Осложнение: {complication}"
        prompt += ". Рекомендации по лечению:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=200)
        recommendation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return recommendation