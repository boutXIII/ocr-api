```bash
python -m venv venv
venv\Scripts\activate  
pip install -r requirements.txt
python -m uvicorn api.main:app --workers 1 --host 0.0.0.0 --port 8080 --log-level trace --reload
```