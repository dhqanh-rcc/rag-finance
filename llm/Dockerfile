FROM python:3.9-slim

WORKDIR /app

COPY llm-service.py .

RUN pip install --no-cache-dir flask google-generativeai requests

EXPOSE 5002

CMD ["python", "llm-service.py"]