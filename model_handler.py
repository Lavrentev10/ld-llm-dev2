import os
import json
import pickle
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_FILE = "data/clinical_recommendations.json"
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
                texts.append(diagnosis)

        embeddings = self.text_model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.index = faiss.IndexFlatIP(self.dimension)
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
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            distances, indices = self.index.search(query_embedding, top_k)

            if indices[0][0] != -1:
                best_match = self.recommendations[indices[0][0]][1]
                similarity = 1 - distances[0][0]
                print(f"Для диагноза \"{diagnosis}\" найдена рекомендация: {best_match} (Сходство: {similarity:.4f})")
            else:
                best_match = "Нет данных"
                print(f"Для диагноза \"{diagnosis}\" нет данных в индексе.")

            results.append(best_match)

        return results


    def generate_combined_recommendation(self, recommendations, diagnoses):
        prompt = f"Диагнозы: {'; '.join(map(str, diagnoses))}\n"

# ======================= Исправить перечисление диагнозов и рекомендаций ========================

        # prompt += "\n".join([f"Рекомендация: {rec}" for rec in recommendations])
        prompt += f"Рекомендации: {'; '.join(map(str,recommendations))}\n"

        prompt += "Осложнения: Артериальная гипертензия\n"

        prompt += """
    Задача: заполнить бланк осмотра пациента. Бланк нужно составлять, используя медицинскую терминологию. Упрощенные слова или жаргонизмы необходимо переформулировать в соответствии с правилами заполнения. Если в тексте встречаются слова или неуместные вставки, не имеющие отношения к медицинскому осмотру, такие участки текста игнорируются. Все поля документа должны быть полностью и логично заполнены.

    Структура бланка осмотра:

    «Жалобы:» – Жалобы необходимо формулировать только в виде симптомов, соответствующих диагнозу, но не повторяющих его напрямую (например: головная боль, слабость, повышение температуры, кашель).

    «Анамнез заболевания:» – Анамнез заболевания составляется с моих слов, но может быть дополнен исходя из жалоб, объективных данных и диагноза. Он должен быть логично структурирован, следуя формату:
    *«Считает себя больным в течение ** дней, с момента, когда ***, связывает заболевание с **».
    При этом не использовать слово «пациент».

    «Объективный осмотр» – Данные заполняются с учетом имеющейся информации, дополнительно указывая объективные симптомы, соответствующие диагнозу.

    «DS:» (Диагноз) – Указывать диагноз в следующем порядке:
        Название
        Этиология
        Степень тяжести
        Осложнения
        Диагноз должен соответствовать классификации и учитывать представленные данные.

    «Код по МКБ-10:» – Указывать код основного диагноза согласно МКБ-10.

    «Рекомендовано:» –
        Рекомендации должны быть сформулированы заново, на основании диагноза, осложнений и анамнеза, с учетом патогенеза и современных клинических рекомендаций.
        Исходный текст рекомендаций использовать нельзя, необходимо перефразировать и дополнить при необходимости.
        Включать:
            Общие рекомендации по режиму (постельный, полупостельный, общий).
            Диету.
            Этиотропную (при возможности), патогенетическую и симптоматическую терапию с указанием конкретных препаратов по МНН (например: «при повышении температуры – парацетамол 500 мг до 4 раз в сутки»).
        Исключить любые логические противоречия и несовместимые назначения.
        В конце указать: «Повторная явка к врачу».

    «Лабораторные обследования:» – Перечислить исследования для уточнения диагноза и оценки состояния. Использовать сокращенные обозначения (ОАК, ОАМ, Б/Х).

    «Инструментальные обследования:» – Указать исследования, необходимые для уточнения диагноза.

    «Консультации:» – Определить специалистов, консультации которых требуются для дифференциальной диагностики и дальнейшего ведения пациента.

    Важно:

    Все параметры должны быть полностью заполнены.
    Если информации недостаточно, добавить логически обоснованные данные, указывая точные характеристики и числовые значения (при необходимости).
    Использовать только актуальные медицинские термины.
    Исключить неуместные вставки и устаревшие выражения.
    Устранить орфографические ошибки, если они есть в исходном тексте.
    Если диагноз можно уточнить по классификации (степень, локализация, течение), сделать это с учетом представленных данных.
        """
        
        
        print("\nПромпт:\n", prompt)
        
        inputs = self.gen_tokenizer(prompt, return_tensors="pt")
        # outputs = self.gen_model.generate(**inputs, max_new_tokens=1000,
        #     do_sample=False, num_beams=1, pad_token_id=self.gen_tokenizer.eos_token_id)

        outputs = self.gen_model.generate(
            **inputs,
            max_new_tokens=2000,  # Увеличьте длину генерации
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            pad_token_id=self.gen_tokenizer.eos_token_id
        )
        
        # response = self.gen_tokenizer.decode(outputs[0],
        #     skip_special_tokens=True).split("Создай единую, непротиворечивую рекомендацию на основе данных выше.")[-1].strip()
        response = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def process_diagnoses(self, diagnoses):

        found_recommendations = self.search_recommendations(diagnoses)
        final_recommendation = self.generate_combined_recommendation(found_recommendations, diagnoses)
        return final_recommendation

if __name__ == "__main__":
    system = MedicalRecommendationSystem()
    if not os.path.exists(INDEX_FILE):
        system.index_data()
    
    input_diagnoses = ["Гипертрофическая кардиомиопатия (ГКМП) с обструкцией ВТЛЖ",
        "Фибрилляция передсердий при ГКМП",]
    


    result = system.process_diagnoses(input_diagnoses)
    print(f"\nИтоговая рекомендация:\n{result}")