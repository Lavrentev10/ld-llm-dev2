from model_handler import ModelHandler

def main():
    model_handler = ModelHandler()

    diagnosis = "Грипп"
    complication = "Пневмония"
    
    recommendation = model_handler.get_recommendation(diagnosis, complication)
    print("Рекомендации по лечению:", recommendation)

if __name__ == "__main__":
    main()