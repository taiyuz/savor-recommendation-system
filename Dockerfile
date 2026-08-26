FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir .

ENV SAVOR_DATA_DIR=/app/data/sample
EXPOSE 8000

CMD ["uvicorn", "savor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
