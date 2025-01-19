'''
Данные для обучения взяты с сайта:
Пожары в России: данные о местах и типах природных пожаров за 2012 – 2021 гг. // МЧС; обработка: Новиков В.А., Тихонов С.В., Инфраструктура научно-исследовательских данных. АНО «ЦПУР»,
2022. Доступ: Лицензия CC BY-SA. Размещено: 31.03.2022. (Ссылка на набор данных:
 https://data.rcsi.science/data-catalog/datasets/202/#dataset-overview )
'''

import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# Функция для отображения процесса выполнения
def log_progress(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

# Шаг 1: Загрузка данных
log_progress("Загрузка данных из файла...")
data_file = "data/big_data.xlsx"  # Укажите путь к вашему файлу Excel
data = pd.read_excel(data_file)
log_progress("Данные успешно загружены.")

# Шаг 2: Предобработка данных
log_progress("Обработка данных...")
data["dt"] = pd.to_datetime(data["dt"], format="%d.%m.%Y")
print(f"Данные загружены. Количество записей: {len(data)}")

# Проверка на пропуски
log_progress("Удаление пропусков...")
data = data.dropna()
log_progress("Пропуски удалены.")

# Шаг 3: Создание GeoDataFrame для географического анализа
log_progress("Создание GeoDataFrame...")
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data["lon"], data["lat"]))
gdf.set_crs(epsg=4326, inplace=True)
log_progress("GeoDataFrame создан.")

# Шаг 4: Кластеризация данных с использованием DBSCAN
log_progress("Кластеризация данных...")
coords = gdf[["lat", "lon"]].values
db = DBSCAN(eps=0.05, min_samples=5).fit(coords)  # Настройка параметров eps и min_samples
gdf["cluster"] = db.labels_
log_progress("Кластеризация завершена.")


log_progress("Оценка качества кластеризации...")

# Убираем шумовые точки для метрик, которые требуют принадлежности к кластеру
core_samples_mask = gdf["cluster"] != -1
coords_core = coords[core_samples_mask]
clusters_core = gdf["cluster"][core_samples_mask]

if len(set(clusters_core)) > 1:  # Проверяем, есть ли больше одного кластера
    # Силуэтный коэффициент
    silhouette_avg = silhouette_score(coords_core, clusters_core)
    log_progress(f"Силуэтный коэффициент: {silhouette_avg:.2f}")

    # Коэффициент Калински-Харабаза
    calinski_harabasz = calinski_harabasz_score(coords_core, clusters_core)
    log_progress(f"Коэффициент Калински-Харабаза: {calinski_harabasz:.2f}")

    # Индекс Дэвиса-Болдина
    davies_bouldin = davies_bouldin_score(coords_core, clusters_core)
    log_progress(f"Индекс Дэвиса-Болдина: {davies_bouldin:.2f}")
else:
    log_progress("Недостаточно кластеров для расчета метрик качества.")

# Шаг 5: Создание карты
log_progress("Создание карты...")
m = folium.Map(location=[gdf["lat"].mean(), gdf["lon"].mean()], zoom_start=5)

# Цвета для типов пожаров
fire_type_colors = {
    "Природный пожар": "red",
    "Лесной пожар": "blue",
    "Контролируемый пал": "green",
    "Неконтролируемый пал": "orange",
    "Торфяной пожар": "purple"
}


# Добавление слоя MarkerCluster для кругов
log_progress("Добавление MarkerCluster для кругов...")
marker_cluster = MarkerCluster(name="Кластеры точек").add_to(m)

# Для каждой точки на карте создаем круг с радиусом 1 км
for idx, row in gdf.iterrows():
    fire_type = row["type_name"]
    color = fire_type_colors.get(fire_type, "gray")  # Назначаем цвет по типу пожара, если тип неизвестен - серый
    folium.Circle(
        location=[row["lat"], row["lon"]],
        radius=300,  # Радиус 300 метров
        color=color,
        fill=True,
        fill_opacity=0.6
    ).add_to(marker_cluster)

log_progress("MarkerCluster с кругами добавлен.")

# Добавление тепловой карты
log_progress("Добавление тепловой карты...")
heatmap_data = [[row["lat"], row["lon"]] for index, row in gdf.iterrows()]
HeatMap(
    heatmap_data,
    radius=25,  # Радиус для создания плавных переходов
    blur=15,    # Радиус размытия
    max_zoom=10,
    gradient={"0.2": 'green', "0.5": 'orange', "0.8": 'red'}
).add_to(folium.FeatureGroup(name="Тепловая карта", show=True).add_to(m))
log_progress("Тепловая карта добавлена.")

# Добавление слоя для кластеров
cluster_layer = folium.FeatureGroup(name="Кластеры", show=True)
for cluster_id in gdf["cluster"].unique():
    if cluster_id == -1:
        continue  # Пропускаем шумовые точки
    cluster_points = gdf[gdf["cluster"] == cluster_id]
    cluster_coords = cluster_points[["lon", "lat"]].values

    # Создаем выпуклую оболочку (convex hull) для точек кластера
    if len(cluster_coords) >= 3:  # Convex hull требует минимум 3 точки
        hull = MultiPoint(cluster_coords).convex_hull
        folium.Polygon(
            locations=[[lat, lon] for lon, lat in hull.exterior.coords],
            color="blue",
            fill=True,
            fill_opacity=0.4,
            popup=f"Кластер {cluster_id}"
        ).add_to(cluster_layer)
    else:
        # Для кластеров с меньшим числом точек добавляем круг
        center = cluster_coords.mean(axis=0)
        folium.Circle(
            location=[center[1], center[0]],
            radius=5000,  # Радиус 5 км
            color="blue",
            fill=True,
            fill_opacity=0.4,
            popup=f"Кластер {cluster_id}"
        ).add_to(cluster_layer)

cluster_layer.add_to(m)

# Шаг 6: Добавление легенды на карту
log_progress("Добавление легенды...")
legend_html = '''
    <div style="position: fixed; 
                bottom: 30px; left: 30px; width: 200px; height: 150px; 
                background-color: white; border:2px solid black; z-index:9999; font-size:14px; padding: 10px;">
        <b>Типы пожаров</b><br>
        <i style="color:red;">&#8226;</i> Природный пожар<br>
        <i style="color:blue;">&#8226;</i> Лесной пожар<br>
        <i style="color:green;">&#8226;</i> Контролируемый пал<br>
        <i style="color:orange;">&#8226;</i> Неконтролируемый пал<br>
        <i style="color:purple;">&#8226;</i> Торфяной пожар<br>
    </div>
'''
m.get_root().html.add_child(folium.Element(legend_html))
log_progress("Легенда добавлена.")

# Добавление панели управления слоями
folium.LayerControl().add_to(m)

# Сохранение карты
map_html = "fires_map_with_circles_and_markercluster.html"
m.save(map_html)
log_progress(f"Карта с кругами и MarkerCluster сохранена в файл {map_html}.")

# Создание аналитики
log_progress("Создание аналитики...")

# Путь для сохранения изображений аналитики
analytics_dir = "analytics"
os.makedirs(analytics_dir, exist_ok=True)

# 1. Распределение по типам пожаров
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x="type_name", palette="Set2")
plt.title("Распределение по типам пожаров")
plt.xlabel("Тип пожара")
plt.ylabel("Количество")
type_count_file = os.path.join(analytics_dir, "type_distribution.png")
plt.savefig(type_count_file)
plt.close()

# 2. Распределение долготы и широты
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="lon", y="lat", hue="type_name", palette="Set1", alpha=0.7)
plt.title("Распределение долготы и широты по типам пожаров")
plt.xlabel("Долгота")
plt.ylabel("Широта")
coord_distribution_file = os.path.join(analytics_dir, "coord_distribution.png")
plt.savefig(coord_distribution_file)
plt.close()

# 3. Распределение по датам
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x="dt", bins=30, kde=True, color="blue")
plt.title("Распределение пожаров по времени")
plt.xlabel("Дата")
plt.ylabel("Количество")
time_distribution_file = os.path.join(analytics_dir, "time_distribution.png")
plt.savefig(time_distribution_file)
plt.close()

# 4. Распределение пожаров по годам
data["year"] = data["dt"].dt.year
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x="year", palette="viridis")
plt.title("Распределение пожаров по годам")
plt.xlabel("Год")
plt.ylabel("Количество пожаров")
year_distribution_file = os.path.join(analytics_dir, "year_distribution.png")
plt.savefig(year_distribution_file)
plt.close()

# 5. Распределение пожаров по месяцам
data["month"] = data["dt"].dt.month
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x="month", palette="coolwarm")
plt.title("Распределение пожаров по месяцам")
plt.xlabel("Месяц")
plt.ylabel("Количество пожаров")
month_distribution_file = os.path.join(analytics_dir, "month_distribution.png")
plt.savefig(month_distribution_file)
plt.close()

# 6. Корреляционная матрица
plt.figure(figsize=(12, 8))
# Включаем все числовые столбцы для построения корреляции
correlation_matrix = data[["lat", "lon", "year", "month", "temperature", "humidity"]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляционная матрица (с температурой и влажностью)")
correlation_matrix_file = os.path.join(analytics_dir, "new_correlation_matrix.png")
plt.savefig(correlation_matrix_file)
plt.close()

# Добавление аналитики в HTML
log_progress("Добавление аналитики в HTML...")

# Отчёт классификации
classification_report_text = f"""
<style>
    .classification-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        align-items: start;
    }}
    .classification-grid img {{
        width: 95%;
        margin: 0 auto;
    }}
    .classification-grid pre {{
        margin: 0 auto;
        padding: 10px;
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 5px;
    }}
    .classification-grid p {{
        text-align: justify;
        margin: 0 20px;
    }}
</style>
<div class="classification-grid">
    <div>
        <p>Модель случайного леса достигла точности <strong>86%</strong>. Лучшие результаты показаны для классов "Лесной пожар" и "Неконтролируемый пал". Хуже всего модель справляется с определением класса "Торфяной пожар" из-за малого количества данных.</p>
        <pre>
                      precision    recall  f1-score   support

  Контролируемый пал       0.77      0.78      0.78     20416
        Лесной пожар       0.94      0.91      0.93     61068
Неконтролируемый пал       0.84      0.84      0.84     30904
     Природный пожар       0.75      0.82      0.78     19568
      Торфяной пожар       0.24      0.54      0.33        95

            accuracy                           0.86    132051
           macro avg       0.71      0.78      0.73    132051
        weighted avg       0.86      0.86      0.86    132051
        </pre>
    </div>
    <div>
        <h3>Матрица ошибок</h3>
        <img src="analytics/confusion_matrix.png" alt="Матрица ошибок">
        <h3>Важность признаков</h3>
        <img src="analytics/feature_importance.png" alt="Важность признаков">
    </div>
</div>
"""

# Основной блок аналитики с визуализациями
analytics_html = f'''
<style>
    .analytics-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        align-items: center;
    }}
    .analytics-grid img {{
        width: 95%;
        margin: 0 auto;
    }}
    .analytics-grid h3 {{
        text-align: center;
    }}
</style>
<div class="analytics-grid">
    <div>
        <h3>Распределение по типам пожаров</h3>
        <img src="{type_count_file}" alt="Распределение по типам пожаров">
    </div>
    <div>
        <h3>Распределение долготы и широты</h3>
        <img src="{coord_distribution_file}" alt="Распределение долготы и широты">
    </div>
    <div>
        <h3>Распределение пожаров по времени</h3>
        <img src="{time_distribution_file}" alt="Распределение пожаров по времени">
    </div>
    <div>
        <h3>Распределение пожаров по годам</h3>
        <img src="{year_distribution_file}" alt="Распределение пожаров по годам">
    </div>
    <div>
        <h3>Распределение пожаров по месяцам</h3>
        <img src="{month_distribution_file}" alt="Распределение пожаров по месяцам">
    </div>
    <div>
        <h3>Корреляционная матрица</h3>
        <img src="{correlation_matrix_file}" alt="Корреляционная матрица">
    </div>
</div>
'''

# Включение аналитики и результатов классификации в общий HTML
html_with_tabs = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Карта и Аналитика</title>
    <style>
        .tabcontent {{
            display: none;
        }}
        .tabcontent.show {{
            display: block;
        }}
        .tablinks {{
            background-color: #f1f1f1;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            font-size: 17px;
        }}
        .tablinks.active {{
            background-color: #ddd;
        }}
    </style>
</head>
<body>
    <div id="coordinates-display">Координаты: -</div>
    <h2>Переключение между картой и аналитикой</h2>
    <button class="tablinks" onclick="openTab(event, 'Map')">Карта</button>
    <button class="tablinks" onclick="openTab(event, 'Analytics')">Аналитика</button>

    <div id="Map" class="tabcontent show">
        <h3>Карта</h3>
        <iframe src="{map_html}" width="100%" height="600px"></iframe>
    </div>

    <div id="Analytics" class="tabcontent">
        <h3>Аналитика</h3>
        {analytics_html}
        {classification_report_text}
    </div>

    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].classList.remove("show");
            }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].classList.remove("active");
            }}
            document.getElementById(tabName).classList.add("show");
            evt.currentTarget.classList.add("active");
        }}
        
    </script>

</body>
</html>
'''

# Сохранение обновленного HTML с аналитикой
tabs_html_file = "map_and_analytics_with_tabs.html"
with open(tabs_html_file, "w") as f:
    f.write(html_with_tabs)
log_progress(f"HTML с аналитикой сохранен в файл {tabs_html_file}.")
