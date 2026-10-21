InJob restart
=============

The ``nvidia_resiliency_ext.fault_tolerance`` package provides section-based
hang detection, worker-failure detection, and coordinated worker restarting
through ``ft_launcher``. The recommended deployment uses Megatron-LM with fixed
section timeouts and NVRx-backed async checkpointing.

Start with the usage guide and the GPT recovery example. A small DDP example
demonstrates section monitoring with synchronous PyTorch checkpoints. Heartbeat
examples are retained as legacy material and are not the recommended integration
path. Automatic timeout calibration is
experimental and is not sufficiently validated for the recommended deployment.

.. toctree::
   :maxdepth: 2
   :caption: InJob restart

   usage_guide
   integration
   examples
   configuration
   deployment
   operations
   services
   timeout_diagnostics
   api
