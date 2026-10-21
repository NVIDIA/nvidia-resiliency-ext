Legacy heartbeat DDP training example
=====================================

Heartbeat monitoring and dynamic calibration are legacy/experimental paths.
For section monitoring with fixed CLI timeouts and the same synchronous PyTorch
checkpoint approach, use :doc:`train_ddp_sections`. For GPT, use
:doc:`megatron_gpt`. The included YAML has no finite initial heartbeat deadline.

.. warning::

   This example loads checkpoints with ``torch.load(..., weights_only=True)``
   because it saves only plain state dictionaries. For PyTorch versions before
   2.10.0, CVE-2026-24747 affects the ``weights_only`` unpickler. Do not load
   untrusted checkpoint files with affected PyTorch versions; use PyTorch
   2.10.0 or newer when checkpoint provenance is not fully trusted.

.. literalinclude:: ../../../../examples/fault_tolerance/train_ddp_heartbeats_api.py
   :language: python
   :linenos:
