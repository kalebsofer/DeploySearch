# Deploying Two Tower Search

## Local Testing

Create virtual environment:

```bash
python -m venv env
source env/bin/activate
```

Install dependencies (python 3.12):

```bash
pip install -r requirements.txt
``` 

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

In a seperate terminal, start the streamlit app:

```bash
streamlit run streamlit_app.py
```

## Local Docker Deployment (broken)

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed on your machine.

## Steps

Navigate to project directory and build Docker Image
   ```bash
   docker build -t twotowersearch .
   ```

Run Docker Container
   ```bash
   docker run -p 8000:8000 twotowersearch
   ```

Open a web browser and navigate to `http://localhost:8000` to access the FastAPI server.

Stop Docker Container

```bash
docker stop <container_id>
```

