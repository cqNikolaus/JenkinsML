FROM python:3.9
WORKDIR /workspace
RUN pip install requests pandas
COPY extract_data.py /workspace/extract_data.py
CMD ["python", "/workspace/data_extraction/extract_data.py"]
