FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt ./
COPY train_model.py ./
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "train_model.py", "--input_csv", "/workspace/build_data.csv", "--output_model", "/workspace/model.pkl"]