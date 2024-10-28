FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/ models/

EXPOSE 8000

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]