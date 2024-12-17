Fault Tolerance
===============

Fault Tolerance is a Python package that features:
   * Workload hang detection.
   * Automatic calculation of timeouts used for hang detection.
   * Detection of rank(s) terminated due to an error.
   * Workload respawning in case of a failure.

The ``nvidia-resiliency-ext`` package also includes the PTL callback ``FaultToleranceCallback`` that simplifies FT package integration with PyTorch Lightning-based workloads.

Fault Tolerance is included in the ``nvidia_resiliency_ext.fault_tolerance`` package.
``FaultToleranceCallback`` is included in the ``nvidia_resiliency_ext.ptl_resiliency`` package.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage_guide
   api
   examples