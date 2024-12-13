FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y curl
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install psycopg2-binary

COPY app/ app/

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "app/streamlit_app.py"]