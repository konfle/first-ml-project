from model_service import ModelService


def main():
    apartment_config = [85, 2015, 2, 20, 1, 1, 0, 0, 1]
    ml_svc = ModelService()
    ml_svc.load_model("rf_v1")
    predict = ml_svc.predict(apartment_config)
    print(f"Cost: {predict}")


if __name__ == "__main__":
    main()
