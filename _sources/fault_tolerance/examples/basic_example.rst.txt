Legacy heartbeat monitoring example
====================================

This older DDP example demonstrates heartbeat monitoring and experimental
timeout calculation. It does **not** save model, optimizer, or training
progress; a worker restart starts training again. Its ``ft_state.json`` holds
calculated monitoring timeouts, not a training checkpoint or launcher YAML.

Heartbeat monitoring is being deprecated as a deployment approach. Use
:doc:`train_ddp_sections` for a small section-based example with training
checkpoint recovery, or :doc:`megatron_gpt` for the recommended GPT workflow.

.. literalinclude:: ../../../../examples/fault_tolerance/basic_ft_example.py
   :language: python
   :linenos:
