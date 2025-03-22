# Используем официальный образ Python как базовый
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . /app

RUN apt update && apt install -y curl
# Обновляем pip и устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Экспонируем порт для Streamlit
EXPOSE 8501

# Команда для запуска Streamlit приложения
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]