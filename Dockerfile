FROM python:3.12.11-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    libjpeg62-turbo \
    libpng16-16 \
    tesseract-ocr \
    tesseract-ocr-rus \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Pillow, обрабатываешь	✅ libjpeg62-turbo, libpng16-16 если нужно будет

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Установка рабочей директории
WORKDIR /code

COPY requirements.txt /code/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /code/

RUN useradd -m user
RUN chown -R user:user /code
USER user

