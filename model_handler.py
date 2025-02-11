import os
import json
import pickle
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_FILE = "clinical_recommendations.json"
INDEX_FILE = "faiss_index.bin"
EMBEDDINGS_FILE = "embeddings.pkl"
MODEL_PATH = "./models/all-MiniLM-L6-v2"
GEN_MODEL_PATH = "./models/Qwen2.5-7B"

if not os.path.exists(MODEL_PATH):
    print("Скачивание SentenceTransformer...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.save(MODEL_PATH)
    print(f"Модель SentenceTransformer установлена в {MODEL_PATH}")

class MedicalRecommendationSystem:
    def __init__(self):
        print("Загрузка моделей...")
        self.text_model = SentenceTransformer(MODEL_PATH)
        self.gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_PATH, trust_remote_code=True)
        self.gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float32)
        
        self.dimension = 384
        self.index = None
        self.recommendations = []
        self._load_index()

    def _load_index(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(EMBEDDINGS_FILE, "rb") as f:
                self.recommendations = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def index_data(self, file_path=DATA_FILE):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.recommendations = []
        texts = []
        
        for entry in data:
            diagnosis = entry["diagnosis"]
            for rec in entry["recommendations"]:
                self.recommendations.append((diagnosis, rec))
                texts.append(f"{diagnosis}: {rec}")
        
        embeddings = self.text_model.encode(texts, convert_to_numpy=True)
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        faiss.write_index(self.index, INDEX_FILE)
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(self.recommendations, f)

    def search_recommendations(self, diagnoses, top_k=1):
        if not self.recommendations:
            print("Индекс пуст")
            return []

        results = []
        for diagnosis in diagnoses:
            query_embedding = self.text_model.encode([diagnosis], convert_to_numpy=True)
            distances, indices = self.index.search(query_embedding, top_k)

            if indices[0][0] != -1:
                best_match = self.recommendations[indices[0][0]][1]
                similarity = distances[0][0]
                print(f"Для диагноза \"{diagnosis}\" найдена рекомендация: {best_match} (Сходство: {similarity:.4f})")
            else:
                best_match = "Нет данных"
                similarity = 0
                print(f"Для диагноза \"{diagnosis}\" нет данных в индексе.")

            results.append(best_match)

        return results

        
        return results

    def generate_combined_recommendation(self, recommendations):

        prompt = "\n".join([f"Рекомендация: {rec}" for rec in recommendations])
        prompt += "\nСоздай единую, непротиворечивую рекомендацию на основе данных выше."
        # print("Промпт:", prompt)
        
        inputs = self.gen_tokenizer(prompt, return_tensors="pt")
        outputs = self.gen_model.generate(**inputs, max_new_tokens=200,
            do_sample=False, num_beams=1, pad_token_id=self.gen_tokenizer.eos_token_id)
        
        response = self.gen_tokenizer.decode(outputs[0],
            skip_special_tokens=True).split("Создай единую, непротиворечивую рекомендацию на основе данных выше.")[-1].strip()

        return response

    def process_diagnoses(self, diagnoses):

        found_recommendations = self.search_recommendations(diagnoses)
        final_recommendation = self.generate_combined_recommendation(found_recommendations)
        return final_recommendation

if __name__ == "__main__":
    system = MedicalRecommendationSystem()
    if not os.path.exists(INDEX_FILE):
        system.index_data()
    
    input_diagnoses = ["Гипертония", "Сахарный диабет"]
    result = system.process_diagnoses(input_diagnoses)
    print(f"\nИтоговая рекомендация:\n{result}")