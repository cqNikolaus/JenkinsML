FROM python:3.9
WORKDIR /app
RUN pip install requests pandas scikit-learn
COPY . /app
CMD ["python", "predict_build_failure.py"]
