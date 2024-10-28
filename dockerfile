FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/ models/
COPY start.sh .

EXPOSE 8000 8501

CMD ["bash", "start.sh"]