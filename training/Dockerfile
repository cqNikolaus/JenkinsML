FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt ./
COPY train_model.py ./
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "train_model.py"]
