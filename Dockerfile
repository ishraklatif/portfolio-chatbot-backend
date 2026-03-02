FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    bash \
    zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pull model at BUILD time so startup is instant
RUN ollama serve & sleep 5 && ollama pull tinyllama && kill %1

COPY . .
RUN chmod +x start.sh

EXPOSE 8000
CMD ["./start.sh"]