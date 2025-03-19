from fastapi import FastAPI, Request, File, UploadFile  # Основные компоненты FastAPI для создания API и работы с файлами
from fastapi.responses import HTMLResponse  # Класс для ответов в виде HTML
from fastapi.templating import Jinja2Templates  # Утилита для рендеринга HTML-шаблонов
from fastapi.middleware.cors import CORSMiddleware  # Middleware для управления политикой CORS
from fastapi.staticfiles import StaticFiles
import uvicorn  # ASGI сервер для запуска приложения
import io  # Ввод-вывод для работы с потоками
import os  # Работа с операционной системой, например, для доступа к переменным окружения
import base64  # Кодирование и декодирование данных в base64
from PIL import Image  # Работа с изображениями
from keras.models import load_model  # Загрузка предобученных моделей Keras
import numpy as np  # Работа с массивами
from typing import Dict  # Типизация для словарей
from pydantic import BaseModel  # Создание моделей данных для валидации
from PIL import ImageOps  # Работа с операциями над изображениями
from fastapi.responses import StreamingResponse  # Класс для потоковой передачи ответов
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt  # построение графиков
import tempfile
import shutil
from datetime import timedelta

app = FastAPI()

# Здесь 'directory' должен указывать на папку, где находятся static файлы
app.mount("/static", StaticFiles(directory="C:\Lesson_FastAPI\static"), name="static")


templates = Jinja2Templates(directory="templates")

# Разрешаем запросы CORS от любого источника
origins = ["*"]  # Для простоты можно разрешить доступ со всех источников
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
  
# Загрузка данных из CSV файла
df = pd.read_csv('C:\Lesson_FastAPI\EUR_USD.csv', sep=';')

# Преобразуем колонку 'Date' в тип datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Сортируем данные по дате (если они не отсортированы)
df = df.sort_values('Date').reset_index(drop=True)

# Задание названий каналов данных
channel_names = ['Price', 'Open', 'Max', 'Min', 'Change']

# Проверка наличия необходимых столбцов
for col in channel_names:
    if col not in df.columns:
        raise ValueError(f"Столбец '{col}' отсутствует в CSV файле.")

# Извлечение данных и преобразование в numpy массив
historical_data = df[channel_names].to_numpy()  # Форма: (6818, 5)

# Извлечение списка дат
historical_dates = df['Date'].tolist()  # Список дат длиной 6818

# Создание словаря с индексами каналов
channel_index = {name: idx for idx, name in enumerate(channel_names)}

# Определение входных и выходных каналов
channel_x = channel_names        # Каналы входных данных ['Price', 'Open', 'Max', 'Min', 'Change']
channel_y = ['Price']            # Канал данных для предсказания ['Price']
SEQ_LEN = 200                    # Длина последовательности (как при обучении модели)

model_TS = load_model('EUR_USD.keras')

   
@app.get("/tseries.html", response_class=HTMLResponse)
async def tseries(request: Request):
    return templates.TemplateResponse("tseries.html", {"request": request})

@app.post("/tseries_predict")
async def tseries_predict(data: Dict[str, str]):

    # Получаем число дней для прогноза
    data_str = data['days']
    
    # Преобразуем строку в целое число
    try:
        n_days = int(data_str)
    except ValueError:
        return {"error": "Неверный формат данных. Пожалуйста, введите целое число от 1 до 60."}
    
    # Проверяем допустимый диапазон
    if n_days < 1 or n_days > 60:
        return {"error": "Пожалуйста, введите число от 1 до 60."}
    
    # Проверяем, что достаточно данных для формирования входной последовательности
    if historical_data.shape[0] < SEQ_LEN:
        return {"error": "Недостаточно исторических данных для предсказания."}
    
    # Получаем индексы входных и выходных каналов
    chn_x = [channel_index[c] for c in channel_x]  # [0, 1, 2, 3, 4]
    chn_y = [channel_index[c] for c in channel_y]  # [0] для 'Price'
    
    # Извлекаем последние SEQ_LEN записей для входной последовательности
    input_seq = historical_data[-SEQ_LEN:, chn_x]  # Форма: (200, 5)
    
    # Проверяем наличие NaN значений и обрабатываем их
    if np.isnan(input_seq).any():
        return {"error": "В данных есть отсутствующие значения. Невозможно выполнить предсказание."}
    
    # Нормализация или масштабирование данных (если это делалось при обучении модели)
    
    scaler = MinMaxScaler()
    input_seq = scaler.fit_transform(input_seq)
    
    # Преобразуем в форму (1, SEQ_LEN, количество признаков)
    input_seq = input_seq.reshape(1, SEQ_LEN, len(chn_x))  # Форма: (1, 200, 5)
    
    predictions = []
    input_sequence = input_seq.copy()
    
    for _ in range(n_days):
        # Делаем предсказание
        pred = model_TS.predict(input_sequence)  # Форма выхода: (1, 1)
        pred_price = pred[0, 0]
        predictions.append(float(pred_price))
        
        # Подготавливаем новый входной временной шаг
        # Используем последние известные значения для других признаков
        last_known_values = input_sequence[0, -1, :].copy()
        price_idx = channel_x.index('Price')
        last_known_values[price_idx] = pred_price  # Обновляем 'Price' на предсказанное значение
        
        # Добавляем новый временной шаг в последовательность
        input_sequence = np.append(input_sequence[:, 1:, :], last_known_values.reshape(1, 1, len(chn_x)), axis=1)
    
    # Подготавливаем даты для предсказаний
    last_date = historical_dates[-1]
    prediction_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]
    
    # Генерируем путь к изображению
    image_path = f'static/predictions_{n_days}.png'
    
    # Строим график
    plot_predictions(historical_dates, historical_data[:, channel_index['Price']], prediction_dates, predictions, image_path)
    
    # Возвращаем путь к изображению и предсказания
    return {
        "image_url": f"/static/{os.path.basename(image_path)}",
        "predictions": predictions,
        "dates": [date.strftime('%Y-%m-%d') for date in prediction_dates]
    }

# Функция для построения графика
def plot_predictions(historical_dates, historical_prices, prediction_dates, predicted_prices, image_path):


    plt.figure(figsize=(12, 6))
    plt.plot(historical_dates, historical_prices, label='Исторические данные')
    plt.plot(prediction_dates, predicted_prices, label='Прогноз', linestyle='--')
    plt.xlabel('Дата')
    plt.ylabel('Курс евро-доллар')
    plt.title('Прогноз курса евро-доллар')
    plt.legend()
    plt.grid(True)
    plt.savefig(image_path)
    plt.close()

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=int(os.environ.get('PORT', 8000)))
