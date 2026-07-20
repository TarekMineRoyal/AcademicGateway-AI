# 🎓 AcademicGateway-AI

> High-performance, localized vector matchmaking and semantic search microservice powering the **AcademicGateway** recommendation engine.

`AcademicGateway-AI` is built with **FastAPI**, **LanceDB**, and local **Nomic-Embed-Text** embeddings accelerated via **PyTorch**. It provides real-time vector synchronization, zero-downtime Blue/Green bulk ingestion, data destruction, and multi-entity semantic matchmaking for students, professors, project blueprints, and technical skills.

---

## 🏗 Architecture Highlights

The microservice is built around clean architectural patterns to guarantee high throughput, strict operational isolation, and protection for hardware resources:

* **CQRS (Command Query Responsibility Segregation)**: Strictly decouples state-changing write operations (`/sync/*`) from read-only semantic queries (`/search/*`).
* **Stateless Searcher Pattern**: Recommendation queries accept flat, enriched text context payloads directly at the API boundary, eliminating real-time SQL join bottlenecks against the main application database.
* **Blue/Green Bulk Synchronization Pipeline**:
  * Bulk ingestion routes (`/sync/bulk/*`) accept large backfill payloads and immediately return an HTTP `202 Accepted` response.
  * Ingestion processes asynchronously in background tasks, slicing incoming arrays into VRAM-safe batches defined by `BATCH_CHUNK_SIZE` (default: `128`) to prevent PyTorch CUDA Out-Of-Memory (OOM) GPU crashes.
  * Bulk data writes to isolated staging tables (`*_sync`). Once processing completes, a zero-downtime PyArrow-based table swap promotes the staging data to production and clears in-memory handle caches.
* **Native Idempotent Deletions**: Deletion endpoints (`DELETE /sync/{domain}/{id}`) bypass the embedding engine entirely, executing direct LanceDB predicate purges to prevent orphaned vector nodes.

---

## 📂 Project Structure

```text
AcademicGateway-AI/
├── api/                        # FastAPI boundary layer
│   ├── routers/                # Endpoint definitions (/search, /sync)
│   ├── dependencies.py         # Dependency injection containers
│   └── main.py                 # FastAPI application entrypoint
├── application/                # Core business logic (CQRS)
│   ├── commands/               # Write side: Bulk sync, single sync, & purges
│   ├── queries/                # Read side: Matchmaking & suggestion logic
│   ├── interfaces/             # Abstract repository & service contracts
│   └── services/formatters/    # Text context builders for embedding models
├── domain/                     # Pure domain entities (Student, Professor, etc.)
├── infrastructure/             # Tech-stack implementations
│   ├── config/                 # Pydantic environment configuration
│   ├── embedding/              # Nomic-Embed-Text & PyTorch driver
│   └── persistence/            # LanceDB vector client & repositories
├── tests/                      # Pytest suite (unit, integration, smoke, api)
├── docs/                       # OpenAPI specifications & schemas
├── requirements.txt            # Project dependencies
└── pytest.ini                  # Test runner configuration
```

---

## ⚙️ Prerequisites & Setup

### Prerequisites
* **Python 3.13+**
* **NVIDIA GPU + CUDA Drivers** (Optional, falls back to CPU automatically via PyTorch if GPU is unavailable)

### Installation

1. **Clone the repository:**
    ```bash
       git clone https://github.com/YourOrg/AcademicGateway-AI.git
       cd AcademicGateway-AI
    ```
2. **Create and activate a virtual environment:**

   * **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

   * **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🎛 Configuration

Environment settings are managed in `infrastructure/config/settings.py` via Pydantic settings. You can override defaults by creating a `.env` file in the project root:

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `LANCE_DB_URI` | `data/vector_space.lance` | Directory path or URI for LanceDB vector storage. |
| `EMBEDDING_MODEL_NAME` | `nomic-ai/nomic-embed-text-v1.5` | Hugging Face model identifier for vector generation. |
| `COMPUTE_DEVICE` | `cuda` | Target compute platform (`cuda` or `cpu`). |
| `BATCH_CHUNK_SIZE` | `128` | Maximum array batch size processed per GPU iteration during bulk sync. |

---

## 🚀 Running the Application

To start the FastAPI server locally with auto-reload enabled:

```bash
uvicorn api.main:app --reload --port 8000
```

> **⏳ First-Run Warmup Note**
> On the initial service boot, PyTorch and Hugging Face Transformers will automatically download the **`nomic-ai/nomic-embed-text-v1.5`** model weights (~500MB) into your local cache directory (`~/.cache/huggingface`). Startup time may take 30–60 seconds depending on network bandwidth. Subsequent startups will initialize instantaneously using local cached weights.

The service will start at `http://localhost:8000`.

* **Live Health Check**: `GET http://localhost:8000/health`
* **Interactive Swagger UI**: `http://localhost:8000/docs`

---

## 🐳 Containerization

For containerized or Kubernetes production environments, the microservice can be packaged via Docker.

### Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose API port
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run Commands

* **CPU Execution (Standard Container):**
  ```bash
  docker build -t academic-gateway-ai .
  docker run -d -p 8000:8000 -e COMPUTE_DEVICE=cpu --name gateway-ai academic-gateway-ai
  ```

* **GPU Execution (NVIDIA CUDA Acceleration):**
  > Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host machine.
  ```bash
  docker run -d -p 8000:8000 --gpus all -e COMPUTE_DEVICE=cuda --name gateway-ai academic-gateway-ai
  ```

---

## 🧪 Running the Test Suite

The repository contains unit, integration, smoke, and API boundary test suites powered by `pytest` and `anyio`.

Execute the entire test suite:

```bash
pytest
```

To run a specific test suite category:

```bash
# API Boundary Tests
pytest tests/api/

# Persistence & Repository Integration Tests
pytest tests/integration/

# CQRS Unit Tests
pytest tests/unit/

# Embeddings Smoke Tests
pytest tests/smoke/
```

---

## 📄 API Documentation

### Interactive Swagger UI
When running locally, FastAPI generates interactive OpenAPI documentation at [http://localhost:8000/docs](http://localhost:8000/docs).
