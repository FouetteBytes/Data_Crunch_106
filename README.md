# AgroChill Forecasting System - Data Crunch Final Round

This project implements a time series forecasting system for predicting fresh vegetable and fruit prices 1 to 4 weeks ahead, incorporating weather data and supporting automated retraining.

**Two primary model versions are available as Docker tags:**
*   `v2.0` (**Recommended, XGBoost**): Utilizes XGBoost models, demonstrating superior accuracy on the evaluation dataset.
*   `v1.0` (LightGBM): Utilizes LightGBM models.

## Overview

Following the legacy of Cornelius Greenvale, the Market King, this system implements the "Freezer Gambit" strategy envisioned by his son Magnus. It uses historical weather patterns and commodity prices to forecast future prices, enabling better decisions on when to sell fresh produce and when to utilize the AgroChill cold storage network.

The core task is to predict weekly fresh prices across various economic centers, one month (4 weeks) ahead. The system includes an API for retrieving forecasts and ingesting new data, and it automatically retrains itself to adapt to new information using the model specified by the Docker tag (XGBoost for v2.0, LightGBM for v1.0).

## Features

*   **Multi-Horizon Forecasting:** Predicts prices 1, 2, 3, and 4 weeks into the future.
*   **Data-Driven:** Uses historical weather and price data.
*   **Machine Learning Models:**
    *   `v2.0`: XGBoost tuned with Optuna.
    *   `v1.0`: LightGBM tuned with Optuna.
*   **Feature Engineering:** Creates time-based, lag, and rolling window features.
*   **RESTful API (FastAPI):**
    *   `GET /api/status`: Check API status, active model type, and retraining info.
    *   `POST /api/predict`: Get price forecasts for the next 4 weeks using the loaded model (implements rolling forecast).
    *   `POST /api/data/weather`: Submit new weather data records.
    *   `POST /api/data/prices`: Submit new price data records.
    *   `POST /api/retrain`: Manually trigger the background retraining process for the loaded model type.
*   **Automated Retraining:** Uses APScheduler to automatically retrain models periodically (default: 24h) incorporating newly submitted data.
*   **Data Persistence:** Incoming data is appended to CSV files (requires volume mounting).
*   **Dockerized:** Containerized for deployment and reproducibility.

## Tech Stack

*   **Language:** Python 3.9+
*   **API Framework:** FastAPI
*   **Data Handling:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn, XGBoost, LightGBM
*   **Hyperparameter Optimization:** Optuna
*   **Scheduling:** APScheduler
*   **Serialization:** Joblib
*   **Containerization:** Docker
*   **Concurrency/Locking:** asyncio, python-filelock
*   **Timezone:** pytz

## Project Structure (Submission Format)

```
/time-series-forecasting
│
├── deployment/                     # Core application and build files
│   ├── Dockerfile                  # Instructions to build the Docker image
│   ├── requirements.txt            # Python dependencies for the XGBoost version
│   └── main.py                     # FastAPI app with XGBoost, retraining, API logic
│
├── data/                           # datasets
│   ├── weather_train_data.csv
│   ├── price_train_data.csv
│   ├── weather_val_data.csv       
│   └── price_val_data.csv         
│
├── saved_models_xgb/               # Trained models
│
├── image_name.txt                  # Contains Docker Hub URI 
├── Documentation.pdf               
├── Presentation/              
└── README.md                       

# Note: Trained model (.joblib) files in saved_models_xgb/ 
# Those are also included INSIDE the Docker image referenced in image_name.txt.
```

## Getting Started

### Prerequisites

*   Python (3.9+ recommended)
*   pip
*   Docker Engine / Docker Desktop
*   Git (for cloning the repository if needed for local dev)

### Installation (Local Development - Assumes XGBoost focus)

*Note: Local development runs `main.py` directly. The primary method for evaluation is intended to be via the provided Docker image.*

1.  **Clone:** `git clone https://github.com/FouetteBytes/time-series-forecasting.git && cd time-series-forecasting`
2.  **Venv:** `python -m venv venv && .\venv\Scripts\activate` (Win) or `source venv/bin/activate` (Mac/Linux)
3.  **Install:** `pip install -r deployment/requirements.txt` (Install from the requirements inside deployment)
4.  **Models:** Ensure the `saved_models_xgb` directory containing pre-trained `.joblib` files exists in the project root. 
5.  **Data:** Ensure the original `.csv` files are in the `data` directory.

## Running the Application

### Using Docker (Recommended & Required for Submission Evaluation)

This uses the pre-built image from Docker Hub, which includes the application and trained models.

1.  **Pull the recommended image:**
    ```bash
    docker pull melkor1/agrochill-app:v2.0
    ```

2.  **Run the container:**
    ```bash
    # Create a local 'data' directory first if it doesn't exist: mkdir data
    # Place original competition CSVs inside this local 'data' directory.
    # Then run:
    docker run -d --name agrochill-container \
      -p 8000:8000 \
      -v "$(pwd)/data":/app/data \
      melkor1/agrochill-app:v2.0
    ```
    *   **`-p 8000:8000`**: **Required.** Maps host:container port. Access API at `http://localhost:8000`.
    *   **`-v "$(pwd)/data":/app/data`**: **Required.** Mounts your local `data` folder (containing original CSVs) into the container at `/app/data`. Incoming data will also be saved here. *(Adjust `$(pwd)` for your OS if needed: `%CD%` for Win CMD, `${PWD}` for PowerShell)*.

3.  **Access API Docs:** `http://localhost:8000/docs`

4.  **Check Logs:** `docker logs agrochill-container`

5.  **Stop Container:** `docker stop agrochill-container`

### Local Development (Uvicorn)

Runs the code from your local `deployment` folder. Useful for debugging. Assumes models are in `../saved_models_xgb` relative to `main.py`.

```bash
# Navigate to the 'deployment' folder first
cd deployment
# Run uvicorn pointing to main:app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# Make sure the ../saved_models_xgb and ../data paths are correct relative to execution
cd .. # Go back to project root when done
```
Access at `http://localhost:8000/docs`. *(Note: Path resolution for models/data might need adjustment in `main.py` if running locally this way vs running from project root).*

## API Usage

See the interactive API documentation (Swagger UI) at `/docs` when running. Key endpoints include `/api/status`, `/api/predict`, `/api/data/weather`, `/api/data/prices`, and `/api/retrain`.

## Retraining

*   Configured for automatic retraining every 24 hours using XGBoost models.
*   Loads original + incoming data from the mounted `/app/data` volume.
*   Re-tunes and re-trains models, updating the live models in memory upon success. Check status via `GET /api/status`.
*   Can be triggered manually via `POST /api/retrain`.

## Model Performance (Evaluation Set RMSE)

*   **v2.0 (XGBoost):**
    *   1w: ~106.6
    *   2w: ~120.6
    *   3w: ~123.8
    *   4w: ~139.8

*   **v1.0 (LightGBM):**
    *   1w: ~136.7
    *   2w: ~126.9
    *   3w: ~151.0
    *   4w: ~162.0

*(XGBoost demonstrated significantly lower RMSE than LightGBM (v1.0) in testing).*
