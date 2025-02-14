FROM python:3.9
WORKDIR /app
RUN pip install requests pandas
COPY extract_data.py /app
CMD ["sh", "-c", "rm -f _build_data.csv build_data.csv && python extract_data.py"]

