# Deploying Two Tower Search

[simplesearchengine.com](https://simplesearchengine.com)

A high-performance semantic search engine built with FastAPI, PyTorch and Faiss. This service provides real-time document search capabilities using a Two-Tower RNN model, see [TwoTowerSearch](https://github.com/kalebsofer/TwoTowerSearch) for more details.

## Overview

The Search Engine API processes natural language queries and returns relevant documents based on semantic similarity. It uses a two-tower neural network architecture for embedding generation and FAISS for efficient similarity search.

![stack](/public/images/stack.png)

The service is containerized using Docker Compose, see below, it is deployed to our own remote server and accessible here: [simplesearchengine.com](https://simplesearchengine.com).

![containers](/public/images/containers.pdf)


## Getting Started

### Prerequisites
- Python 3.11+
- Docker
- Docker Compose
- Git

### Local Development

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/DeploySearch.git
    cd DeploySearch
    ```

2. Create and activate virtual environment:
    ```bash
    # For Windows
    python -m venv env
    .\env\Scripts\Activate.ps1

    # For Unix/Linux
    python -m venv env
    source env/bin/activate
    ```

3. Install dependencies:
    ```bash
    cd frontend
    pip install -r requirements.txt
    ```

4. Run the development server:
    ```bash
    python -m frontend.app.run
    ```

### Docker Development

1. Build and run using Docker Compose:
    ```bash
    # Development environment
    docker-compose -f docker-compose.dev.yml up --build
    ```

2. Run automated tests:
    ```bash
    ./test_local.sh
    ```

## API Reference

### Endpoints

#### GET /
Health check endpoint

**Response**
```json
{
    "message": "Welcome to the search API. Use POST /search to send queries."
}
