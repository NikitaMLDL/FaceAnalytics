# Используем официальный образ Python как базовый
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . /app
# Копируем vector_db_service, так как он используется внутри backend

RUN apt update && apt install -y curl
# Обновляем pip и устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Экспонируем порт для FastAPI
EXPOSE 8000

# Команда для запуска FastAPI приложения с помощью Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
