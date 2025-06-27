FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .[web]
CMD ["sentiment-web", "--model", "model.joblib", "--host", "0.0.0.0"]
