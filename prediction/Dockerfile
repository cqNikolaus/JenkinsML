FROM python:3.9-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas scikit-learn
COPY predict.py ./
COPY train_model.py ./
CMD ["python", "predict.py"]