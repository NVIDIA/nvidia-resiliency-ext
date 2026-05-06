Section API usage example with DDP
==================================

.. warning::

   This example loads checkpoints with ``torch.load(..., weights_only=True)``
   because it saves only plain state dictionaries. For PyTorch versions before
   2.10.0, CVE-2026-24747 affects the ``weights_only`` unpickler. Do not load
   untrusted checkpoint files with affected PyTorch versions; use PyTorch
   2.10.0 or newer when checkpoint provenance is not fully trusted.

.. literalinclude:: ../../../../examples/fault_tolerance/train_ddp_sections_api.py
   :language: python
   :linenos:
