nvidia-resiliency-ext
=====================

**nvidia-resiliency-ext** is a set of tools developed by NVIDIA to improve large-scale distributed training resiliency.

.. image:: ./media/nvrx_docs_source.png
   :width: 750
   :alt: Figure highlighting core NVRx features including automatic restart, hierarchical checkpointing, fault detection and health checks.

Features
--------

* `Hang detection and automatic in-job restarting <fault_tolerance/index.html>`_
* `In-process restarting <inprocess/index.html>`_
* `Async checkpointing <checkpointing/async/index.html>`_
* `Local checkpointing <checkpointing/local/index.html>`_
* `Straggler (slower ranks) detection <straggler_det/index.html>`_
* `Shared utilities and distributed logging <shared_utils/index.html>`_

.. toctree::
   :maxdepth: 3
   :caption: Documentation contents:

   fault_tolerance/index
   inprocess/index
   checkpointing/async/index
   checkpointing/local/index
   straggler_det/index
   shared_utils/index
