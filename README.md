# Market Intelligence Dashboard

> **Elevator Pitch**: A high-performance, quantitative sector rotation engine that combines a real-time Streamlit dashboard for visual analysis with an async FastAPI backend for algorithmic trading integration.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)
![Docker](https://img.shields.io/badge/docker-ready-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)

---

## 🏗️ Architecture

The project employs a **Hybrid Architecture** designed for scalability and code reuse. The frontend and backend are decoupled services that share a common core library (`src/`) and data volume, ensuring a Single Source of Truth for all financial logic.

```mermaid
graph TD
    subgraph "Core Logic (src/)"
        A[Data Engine] --> B[Features & Signals]
        B --> C[Vectorized Scorer]
        C --> D[Backtest Engine]
    end

    subgraph "Services"
        UI[Streamlit Dashboard<br>(app.py)]
        API[FastAPI Backend<br>(api/)]
    end

    subgraph "Data Layer"
        Cache[(Parquet Cache<br>/data)]
    end

    UI --> A
    API --> A
    A <--> Cache
```

- **Frontend (`app.py`)**: Interactive Streamlit dashboard for research, backtesting, and report generation.
- **Backend (`api/`)**: High-throughput FastAPI service for programmatic access and external integrations.
- **Core (`src/`)**: Shared library containing all business logic (data fetching, signal generation, backtesting).
- **Data**: Shared Docker volume for persistent caching of market data.

---

## ✨ Features

### 📊 Dashboard Capabilities (Streamlit)
*   **Sector Rotation Strategy**: Real-time scoring of 11 GICS sectors based on Momentum, Volatility, Breadth, and Liquidity.
*   **Visual Backtesting**: Interactive simulations with customizable parameters (rebalance frequency, transaction costs).
*   **Scenario Analysis**: specific stress-testing tools (e.g., "What if Volatility spikes 50%?").
*   **Automated Reporting**: One-click generation of investment memos and CSV exports.

### ⚡ API Capabilities (FastAPI)
*   **Programmatic Access**: Fetch live sector scores and signals via REST.
*   **Algo-Trading Triggers**: Webhook-ready endpoints for automated execution bots.
*   **Compute Offloading**: Offload heavy backtest simulations to the backend server.
*   **Async I/O**: Non-blocking architecture for high-concurrency requests.

---

## 🚀 Quick Start (Docker)

The preferred way to run the full stack is via Docker Compose.

1.  **Start the Stack**
    ```bash
    docker-compose up --build
    ```

2.  **Access Services**
    *   **Dashboard**: [http://localhost:8501](http://localhost:8501)
    *   **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

3.  **Stop**
    ```bash
    docker-compose down
    ```

---

## 🛠️ Local Development

For contributors who prefer running locally without Docker:

1.  **Environment Setup**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Run Dashboard**
    ```bash
    streamlit run app.py
    ```

3.  **Run API**
    ```bash
    uvicorn api.main:app --reload --port 8000
    ```

---

## 📡 API Documentation

Access the full interactive Swagger UI at `/docs`. Key endpoints include:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/market/snapshot` | Get latest market snapshot and sector data. |
| `POST` | `/analysis/score` | Compute sector scores with custom weights. |
| `POST` | `/backtest/run` | Run a full vectorized backtest simulation. |
| `GET` | `/health` | Service health check. |

---

## ⚙️ Configuration

Tune the model via `config.py`. Key parameters:

*   `WEIGHT_PRESETS`: Define custom scoring mixes (Momentum vs. Volatility focus).
*   `MOMENTUM_LOOKBACK`: Default lookback window for return calculations (e.g., 21 days).
*   `SECTOR_ETFS`: List of ticker symbols for the sector universe.
*   `SNAPSHOT_DIR`: Path for caching real-time data.

---

## 📂 Project Structure

```text
├── api/                 # FastAPI Backend
│   ├── routes/          # API Endpoints (market_data, analysis, backtest)
│   ├── services.py      # Business logic interface
│   └── main.py          # App entry point
├── src/                 # Shared Core Library
│   ├── data_engine.py   # Data fetching & caching (TradingView/YFinance)
│   ├── features.py      # Signal generation & feature engineering
│   ├── backtest.py      # Vectorized backtest engine
│   └── scorer.py        # Composite scoring logic
├── app.py               # Streamlit Frontend
├── config.py            # Global Configuration
├── data/                # Local data cache (Parquet)
├── Dockerfile           # Multi-stage build
└── docker-compose.yml   # Orchestration
```
