FROM python:3.11-slim

# Graphviz (the real program) is required for the UML diagrams.
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "main.py"]
