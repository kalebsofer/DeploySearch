# Deploying Two Tower Search

## Local Testing

Create virtual environment:

```bash
python -m venv env
source env/bin/activate
```

Install dependencies (python 3.12):

```bash
cd frontend
pip install -r requirements.txt
``` 

Start the FastAPI server:

```bash
python -m frontend.app.run
```

In a seperate terminal, start the streamlit app:

```bash
cd frontend/app
streamlit run streamlit_app.py
```

