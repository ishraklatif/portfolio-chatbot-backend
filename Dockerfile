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

# Pull model at build time using bash explicitly
RUN bash -c "ollama serve & sleep 5 && ollama pull qwen2:0.5b; kill %1 || true"

COPY . .
RUN chmod +x start.sh

EXPOSE 8000
CMD ["./start.sh"]