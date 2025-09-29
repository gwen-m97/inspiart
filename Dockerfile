  FROM python:3.10.18-bookworm
  RUN pip install --upgrade pip
  COPY api api
  COPY models/model_Xception_alldata_finetuned.keras models/model_Xception_alldata_finetuned.keras
  COPY models/model_clip models/model_clip
  COPY requirements.txt requirements.txt
  RUN pip install -r requirements.txt
  CMD uvicorn api.fast:app --host 0.0.0.0 --port=$PORT
