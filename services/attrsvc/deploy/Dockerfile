# NVRX Attribution Service - Docker Image
#
# Build (from repo root):
#   docker build -t nvrx-attrsvc -f services/attrsvc/deploy/Dockerfile .
#
# Stamp the build so /healthz can identify it:
#   docker build -t nvrx-attrsvc -f services/attrsvc/deploy/Dockerfile \
#       --build-arg NVRX_PACKAGE_VERSION="$(git describe --tags --always --dirty)" \
#       --build-arg NVRX_BUILD_TAG="main-$(git rev-parse --short=8 HEAD)" .
#
# Run:
#   docker run -d \
#       -p 8000:8000 \
#       -e NVRX_ATTRSVC_ALLOWED_ROOT=/data/logs \
#       -e LLM_API_KEY_FILE=/run/secrets/llm_api_key \
#       -v /path/to/logs:/data/logs:ro \
#       -v /secure/llm_api_key:/run/secrets/llm_api_key:ro \
#       nvrx-attrsvc
#
# Build from repo root to include nvidia_resiliency_ext library

FROM python:3.12-slim-bookworm

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (curl for healthcheck)
# APT::Sandbox::User=root because the image is built rootless on hosts with no
# /etc/subuid range, where only one UID is mapped and apt cannot drop to _apt.
# The sandbox guards against untrusted repositories; this build uses the
# official Debian repos of the base image and already runs as root.
RUN apt-get -o APT::Sandbox::User=root update && \
    apt-get -o APT::Sandbox::User=root install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install the standalone service dependencies independently from the root
# package's workload-side dependency set.
COPY services/attrsvc/deploy/requirements.txt ./services/attrsvc/deploy/requirements.txt
RUN pip install --no-cache-dir -r services/attrsvc/deploy/requirements.txt

# Install the root package code without pulling workload-side dependencies.
COPY pyproject.toml README.md build.py ./
COPY src/ ./src/
ENV STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1
ARG NVRX_PACKAGE_VERSION=0.0.0
RUN POETRY_DYNAMIC_VERSIONING_BYPASS="${NVRX_PACKAGE_VERSION}" \
    pip install --no-cache-dir --no-deps .

# Baked so /healthz can say what is running. Without it every image reports
# the same version and a rollout cannot be confirmed from outside the host.
# Overridable at run time by setting NVRX_ATTRSVC_BUILD_TAG on the container.
ARG NVRX_BUILD_TAG=""
ENV NVRX_ATTRSVC_BUILD_TAG=${NVRX_BUILD_TAG}

# Deployments mount a log directory here. The service logs to stderr, which the
# container runtime keeps on the host; writing a copy to the mount makes startup
# state readable by anyone who can see the shared filesystem. Override or clear
# NVRX_ATTRSVC_LOG_FILE to opt out.
ENV NVRX_ATTRSVC_LOG_FILE=/var/log/nvrx/attrsvc.log

# Verify the service command and its import graph without PyTorch.
RUN command -v nvrx-attrsvc && \
    python -c "import nvidia_resiliency_ext.services.attrsvc.app" && \
    python -c "import importlib.util; assert importlib.util.find_spec('torch') is None"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:8000/healthz || exit 1

# Default environment
ENV NVRX_ATTRSVC_HOST=0.0.0.0
ENV NVRX_ATTRSVC_PORT=8000

EXPOSE 8000

CMD ["nvrx-attrsvc"]
