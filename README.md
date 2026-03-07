# Trademarkia Semantic Search API

This repository contains a high-performance Semantic Search API powered by **FastAPI**, **ChromaDB**, and **SentenceTransformers**. It implements an advanced **Cluster-Aware Semantic Cache** using a Gaussian Mixture Model (GMM) to intelligently route queries and significantly reduce database I/O, resulting in lightning-fast search responses.

## Core Architecture

The system is built on a custom data pipeline:
1. **Data Preprocessing** (`pipeline/process_all.py`): Iterates through the raw dataset (20 Newsgroups), cleans the text extracts, and saves them to `data/complete_preprocessing/`.
2. **Vector Ingestion** (`pipeline/setup_chromadb.py`): Embeds the documents using `bge-small-en-v1.5` and ingests them into a persistent local ChromaDB using batching for stability.
3. **Fuzzy Clustering** (`pipeline/clustering.py`): Trains a Gaussian Mixture Model (GMM) on the embeddings to discover latent semantic clusters (k=125) and tags the ChromaDB vectors with fuzzy probability distributions.
4. **Semantic Search & Cache Service** (`app/main.py` & `app/cache_logic.py`): A FastAPI application that vectorizes incoming user queries, predicts the query's semantic cluster via the GMM, and checks an in-memory Cluster-Aware Semantic Cache. 
   - **Cache Hit**: Resolves directly from RAM for sub-millisecond latency.
   - **Cache Miss**: Filters the ChromaDB query to only search within the target cluster, returning the results and updating the cache.

---

## 🚀 Setup and Installation

### Prerequisites
- **Python 3.10+**
- **Docker** and **Docker Compose** (for containerized deployment)

### Local Development Setup

1. **Clone the repository and enter the directory**:
   ```bash
   cd trademarkia_task
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

*(Note: Data generation scripts may require additional dependencies like `scikit-learn` for GMM training).*

---

## 🛠 Running the Pipeline (Data Preparation)

If you are starting from scratch and need to rebuild the database and cluster models, follow these steps in order:

1. **Preprocess Dataset**:
   ```bash
   python pipeline/process_all.py
   ```
2. **Ingest embeddings into ChromaDB**:
   Ensure you have your embeddings (`data/corpus_embeddings.npy`) and mapping paths (`data/corpus_paths.txt`) ready, then run:
   ```bash
   python pipeline/setup_chromadb.py
   ```
3. **Train GMM and Tag Clusters**:
   ```bash
   python pipeline/clustering.py
   ```
   *This trains the GMM and saves `app/models/gmm_model.pkl`.*

---

## 🐳 Running the API Service

### Option 1: Using Docker Compose (Recommended)

The easiest way to run the service is using Docker. The provided `docker-compose.yml` builds the FastAPI service and mounts the localized `chroma_db/` directory so data remains persistent.

1. **Start the containers** in detached mode:
   ```bash
   docker compose up -d
   ```
2. **Stop the containers**:
   ```bash
   docker compose down
   ```

The API will be accessible at `http://localhost:8000`.

### Option 2: Running Locally natively
If you prefer running the uvicorn server directly on your host machine:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 📖 API Endpoints

Once the application is running, you can access the automatically generated interactive API documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Core Endpoints:

- `POST /query`: Submits a semantic search query.
  **Request Body**:
  ```json
  {
    "query": "artificial intelligence advances",
    "top_k": 5
  }
  ```
- `GET /cache/stats`: Returns current real-time statistics of the cluster-aware semantic cache.
- `DELETE /cache`: Flushes the entire in-memory cache and resets stats.
