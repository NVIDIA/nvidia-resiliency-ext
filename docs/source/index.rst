nvidia-resiliency-ext v0.2.0
============================

**nvidia-resiliency-ext** is a set of tools developed by NVIDIA to improve large-scale distributed training resiliency.

Features
--------

* Hang detection and automatic in-job restarting
* In-process restarting
* Async checkpointing
* Local checkpointing
* Straggler (slower ranks) detection

.. toctree::
   :maxdepth: 3
   :caption: Documentation contents:

   fault_tolerance/index
   inprocess/index
   checkpointing/async/index
   checkpointing/local/index
   straggler_det/index
