FROM python:3.9
WORKDIR /app
RUN pip install requests pandas
COPY . /app
CMD ["python", "extract_data.py"]
