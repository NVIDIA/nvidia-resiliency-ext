# NVRX Attribution Service - Docker Image
#
# Build (from repo root):
#   docker build -t nvrx-attrsvc -f services/attrsvc/deploy/Dockerfile .
#
# Stamp the build so /healthz can identify it:
#   docker build -t nvrx-attrsvc -f services/attrsvc/deploy/Dockerfile \
#       --build-arg NVRX_PACKAGE_VERSION="0.5.0.dev+$(git rev-parse --short=8 HEAD)" \
#       --build-arg NVRX_BUILD_TAG="$(git describe --tags --always --dirty)" .
#
# NVRX_PACKAGE_VERSION has to be a valid package version: it is passed through
# to POETRY_DYNAMIC_VERSIONING_BYPASS, and between releases "git describe"
# yields something like v0.5.0-10-gabcdc9c, which the install then rejects.
# NVRX_BUILD_TAG is free-form, so the readable git description belongs there.
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
# Site settings -- the allowed root, the model credential, Slack -- may also be
# put in /var/lib/nvrx/attrsvc.env, which the service reads at startup; see
# attrsvc.env.example. Anything passed with -e or --env-file wins over it.
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
# package's workload-side dependency set. --no-deps below drops every declared
# dependency, including the ones attrsvc imports at runtime, so anything the
# service imports has to be listed here too; test_attrsvc_service_requirements
# fails if this file and pyproject disagree.
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

# Verify the service command and its import graph without PyTorch.
RUN command -v nvrx-attrsvc && \
    python -c "import nvidia_resiliency_ext.services.attrsvc.app" && \
    python -c "import importlib.util; assert importlib.util.find_spec('torch') is None"

# Health check. Build with docker, or podman --format docker: podman's default
# OCI format drops HEALTHCHECK silently, leaving an image with no probe at all.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:8000/healthz || exit 1

# The bind address is a property of running in a container, not a site
# preference: the service defaults to 127.0.0.1, which is right for a host
# install and useless in a namespace, where published ports would reach a
# listener answering only inside it. The port is baked alongside it so EXPOSE
# and the health check agree with what the service actually listens on. A
# deployment that needs a different port overrides both.
ENV NVRX_ATTRSVC_HOST=0.0.0.0
ENV NVRX_ATTRSVC_PORT=8000

EXPOSE 8000

CMD ["nvrx-attrsvc"]
