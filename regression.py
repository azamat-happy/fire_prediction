'''
Данные для обучения взяты с сайта:
Пожары в России: данные о местах и типах природных пожаров за 2012 – 2021 гг. // МЧС; обработка: Новиков В.А., Тихонов С.В., Инфраструктура научно-исследовательских данных. АНО «ЦПУР»,
2022. Доступ: Лицензия CC BY-SA. Размещено: 31.03.2022. (Ссылка на набор данных:
 https://data.rcsi.science/data-catalog/datasets/202/#dataset-overview )
'''

import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np

# Функция для отображения процесса выполнения
def log_progress(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

# Инициализация LabelEncoder
label_encoder = LabelEncoder()

# Шаг 1: Загрузка данных
log_progress("Загрузка данных из файла...")
data_file = "data/big_data.xlsx"  # Укажите путь к вашему файлу Excel
data = pd.read_excel(data_file)
log_progress("Данные успешно загружены.")

# Шаг 2: Предобработка данных
log_progress("Обработка данных...")
data['dt'] = pd.to_datetime(data['dt'], format='%Y-%m-%d %H:%M:%S')

# Создаем столбцы 'year' и 'month'
data['year'] = data['dt'].dt.year
data['month'] = data['dt'].dt.month

# Удаление пропусков
data = data.dropna()
log_progress("Пропуски удалены.")

# Шаг 3: Подготовка данных
X = data[["temperature", "humidity", "lat", "lon", "year", "month"]]  # Признаки
y = data["type_name"]  # Целевая переменная

# Преобразуем категориальные данные в числовые
y_encoded = label_encoder.fit_transform(y)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Сохраняем тестовые данные для использования в будущем
test_data_dir = "test_data"
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)
np.save(os.path.join(test_data_dir, "X_test.npy"), X_test)
np.save(os.path.join(test_data_dir, "y_test.npy"), y_test)

# Балансировка классов
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
log_progress("Балансировка классов завершена.")

# Шаг 4: Обучение модели
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_res, y_res)
log_progress("Модель обучена.")

# Проверка и создание директории models
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Сохраняем модель и LabelEncoder
joblib.dump(rf_model, os.path.join(models_dir, "random_forest_model.pkl"))
joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.pkl"))
log_progress("Модель и LabelEncoder сохранены.")
