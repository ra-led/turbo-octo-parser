FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
COPY . .
CMD ["python", "worker.py"]

