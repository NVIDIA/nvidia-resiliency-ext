nvidia-resiliency-ext
=====================

**nvidia-resiliency-ext** is a set of tools developed by NVIDIA to improve large-scale distributed training resiliency.

Features
--------

* `InJob restart with section-based monitoring <fault_tolerance/index.html>`_
* `Async checkpointing <checkpointing/async/index.html>`_
* `Local checkpointing <checkpointing/local/index.html>`_
* `Straggler (slower ranks) detection <straggler_det/index.html>`_
* `Failure attribution <attribution/index.html>`_
* `Shared utilities and distributed logging <shared_utils/index.html>`_

.. toctree::
   :maxdepth: 3
   :caption: Documentation contents:

   release-notes
   fault_tolerance/index
   checkpointing/async/index
   checkpointing/local/index
   straggler_det/index
   attribution/index
   shared_utils/index
