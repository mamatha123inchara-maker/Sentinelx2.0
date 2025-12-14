FROM python:3.10-slim

WORKDIR /app

COPY Solar_pipeline/Docker/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "Solar_pipeline/pipeline.py/pipeline.py"]
