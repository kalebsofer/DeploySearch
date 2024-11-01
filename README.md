# Deploying Two Tower Search

### Website: [simplesearchengine.com](https://simplesearchengine.com)

A high-performance semantic search engine built with FastAPI, PyTorch and Faiss. This service provides real-time document search capabilities using Two-Tower RNN model, see [TwoTowerSearch](https://github.com/kalebsofer/TwoTowerSearch) for more details.

## Overview

This Search Engine API processes natural language queries and returns relevant documents based on semantic similarity. It uses a two-tower neural network architecture for embedding generation and FAISS for efficient similarity search.

### Stack

![stack](/public/images/stack.png)

### Architecture

The service is containerized using Docker Compose, it is deployed to a Hetzner remote server.
![containers](/public/images/containers.png)


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
3. *Make some changes*

#### Local deployment 
1. Create `.env.dev` file by copying `.env.example`.
2. Create data and model files in your local directory under minio/data/
3. Build and deploy locally, if this is your first time building it will take some time.

    ```bash
    docker-compose -f docker-compose.dev.yml --env-file .env.dev up --build
    ```

#### Production deployment
1. ssh into the production server
2. Create `.env.prod` file by copying `.env.example`.
3. Transfer docker-compose.prod.yml file to the production server.
    ```bash
    scp docker-compose.prod.yml root@<server-ip>:/root/docker-compose.prod.yml
    ```
4. Create `.env.prod` file by copying `.env.example`.
5. Pull images from docker hub 

6. Deploy the containers

    ```bash
    docker-compose -f docker-compose.prod.yml --env-file .env.prod up --build
    ```