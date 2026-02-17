# NVRX Attribution Service - Docker Image
#
# Build (from repo root):
#   docker build -t nvrx-attrsvc -f services/nvrx_attrsvc/deploy/Dockerfile .
#
# Run:
#   docker run -d \
#       -p 8000:8000 \
#       -e NVRX_ATTRSVC_ALLOWED_ROOT=/data/logs \
#       -e NVIDIA_API_KEY=nvapi-... \
#       -v /path/to/logs:/data/logs:ro \
#       nvrx-attrsvc
#
# Build from repo root to include nvidia_resiliency_ext library

FROM python:3.11-slim

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (curl for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install main library (no CUDA extensions needed for attribution)
COPY pyproject.toml ./
COPY src/ ./src/
ENV STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1
RUN pip install --no-cache-dir --no-deps .

# Copy and install attribution service
COPY services/pyproject.toml ./services/
COPY services/nvrx_attrsvc/ ./services/nvrx_attrsvc/
COPY services/nvrx_smonsvc/ ./services/nvrx_smonsvc/
RUN pip install --no-cache-dir ./services

# Verify commands are available
RUN command -v nvrx-attrsvc && command -v nvrx-smonsvc

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:8000/healthz || exit 1

# Default environment
ENV NVRX_ATTRSVC_HOST=0.0.0.0
ENV NVRX_ATTRSVC_PORT=8000

EXPOSE 8000

# Default to attribution service; override with nvrx-smonsvc for monitor
CMD ["nvrx-attrsvc"]
