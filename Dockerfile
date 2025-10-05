FROM python:3.12.11-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    libjpeg62-turbo \
    libpng16-16 \
    tesseract-ocr \
    tesseract-ocr-rus \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Установка рабочей директории
WORKDIR /code

# Копируем и устанавливаем зависимости (кешируется отдельным слоем)
COPY requirements.txt /code/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . /code/

RUN useradd -m user
RUN chown -R user:user /code
USER user

