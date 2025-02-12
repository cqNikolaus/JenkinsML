FROM python:3.9
WORKDIR /workspace
RUN pip install requests pandas
COPY . /workspace
CMD ["python", "/workspace/extract_data.py"]
