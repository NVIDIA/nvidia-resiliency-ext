ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /opt/nvrx

RUN mkdir -p /opt/nvrx/src/nvidia_resiliency_ext/services

COPY src/nvidia_resiliency_ext/__init__.py /opt/nvrx/src/nvidia_resiliency_ext/__init__.py
COPY src/nvidia_resiliency_ext/services/__init__.py /opt/nvrx/src/nvidia_resiliency_ext/services/__init__.py
COPY src/nvidia_resiliency_ext/services/nodestatesvc /opt/nvrx/src/nvidia_resiliency_ext/services/nodestatesvc

ENV PYTHONPATH=/opt/nvrx/src

EXPOSE 8000
ENTRYPOINT ["python", "-m", "nvidia_resiliency_ext.services.nodestatesvc"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
