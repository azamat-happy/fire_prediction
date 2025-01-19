'''
Данные для обучения взяты с сайта:
Пожары в России: данные о местах и типах природных пожаров за 2012 – 2021 гг. // МЧС; обработка: Новиков В.А., Тихонов С.В., Инфраструктура научно-исследовательских данных. АНО «ЦПУР»,
2022. Доступ: Лицензия CC BY-SA. Размещено: 31.03.2022. (Ссылка на набор данных:
 https://data.rcsi.science/data-catalog/datasets/202/#dataset-overview )
'''

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Загружаем модель и LabelEncoder
model = joblib.load("models/random_forest_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Вводим данные для предсказания
# температура, влажность, широта, долгота, год, месяц
new_data = np.array([[5.99235199200885, 70.1386822547276, 59.1004, 54.3478, 2012, 5]])

# Преобразуем в pandas DataFrame с именами признаков
columns = ["temperature", "humidity", "lat", "lon", "year", "month"]
new_data_df = pd.DataFrame(new_data, columns=columns)

# Делаем предсказание
prediction_encoded = model.predict(new_data_df)
prediction = label_encoder.inverse_transform(prediction_encoded)

# Получаем вероятности для каждого класса
prediction_proba = model.predict_proba(new_data_df)

# Преобразуем строковое предсказание в его числовой код
predicted_class_index = label_encoder.transform([prediction[0]])[0]

# Получаем вероятность для предсказанного класса
prediction_confidence = prediction_proba[0][predicted_class_index]

# Выводим результаты
print(f"Предсказание для новых данных: {prediction[0]}")
print(f"Точность предсказания: {prediction_confidence*100:.2f}%")