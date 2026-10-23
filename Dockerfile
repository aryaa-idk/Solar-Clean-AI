# ================================================================
# SolarClean AI — Multi-Arch Production Deployment Image
# ================================================================
# Builds the single monolithic Python server that serves BOTH:
#   - REST API  (/api/*)
#   - Static frontend (dashboard/)
FROM python:3.12-slim-bookworm

# Avoid buffering & enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source + data
COPY . .

# Default configuration (overridable via docker run / compose env)
ENV HOST=0.0.0.0 \
    PORT=8085

EXPOSE 8085

# Pre-export the static dashboard bundle (non-fatal if weather API unreachable)
# then launch the production HTTP server
CMD ["python", "solarclean_server.py"]
