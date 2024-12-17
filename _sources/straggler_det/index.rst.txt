Straggler Detection
===================

The **Straggler Detection** package's purpose is to detect slower ranks participating in a PyTorch distributed workload.
The ``nvidia-resiliency-ext`` package also includes the PTL callback ``StragglerDetectionCallback`` that simplifies integration with PyTorch Lightning-based workloads.

Straggler Detection is included in the ``nvidia_resiliency_ext.straggler`` package.
``StragglerDetectionCallback`` is included in the ``nvidia_resiliency_ext.ptl_resiliency`` package.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage_guide
   api
   examples