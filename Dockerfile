  FROM python:3.10.18-bookworm
  RUN pip install --upgrade pip
  COPY api api
  COPY requirements.txt requirements.txt
  RUN pip install -r requirements.txt
  CMD uvicorn api.fast:app --host 0.0.0.0 --port=$PORT
