FROM python:3.9-slim

WORKDIR /app

COPY faiss-service.py .
COPY all_stocks_5yr.csv .

RUN pip install --no-cache-dir \
    flask \
    pandas \
    langchain[openai] \
    faiss-cpu \
    transformers \
    torch

EXPOSE 5001

CMD ["python", "faiss-service.py"]