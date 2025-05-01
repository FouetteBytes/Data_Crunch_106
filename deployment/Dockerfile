FROM python:3.9-slim AS base

LABEL version="2.0-xgboost"
LABEL description="AgroChill Crop Price Prediction API using XGBoost"

# environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC

# working  directory
WORKDIR /app

# python dependencies
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# application Image
FROM base AS final

WORKDIR /app

COPY deployment/ /app/
COPY saved_models_xgb/ /app/saved_models_xgb/
COPY data/ /app/data/ 


RUN mkdir -p /app/data && \
    chown -R 65534:65534 /app/data || true 

# expose the port the app runs on 
EXPOSE 8000

# command to run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]