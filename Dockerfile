FROM python:3.13-slim

# libgomp1 est requis par faiss-cpu (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copie de l'exécutable uv depuis l'image officielle
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Installer les dépendances en premier (couche mise en cache séparément)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copier le code source
COPY . .

# Créer les répertoires de données s'ils n'existent pas encore
RUN mkdir -p vector_db database data

# Commande par défaut : API (surchargée par docker-compose pour les autres services)
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
