Examples
========

Choose an example by the recovery behavior you want to exercise. Monitoring
and checkpoint storage are separate choices: the small DDP example keeps
synchronous PyTorch saves, while GPT uses NVRx async saves through Megatron.

.. toctree::
   :maxdepth: 2
   :caption: Recommended examples

   examples/megatron_gpt
   examples/train_ddp_sections
   examples/singleton_deployment

.. toctree::
   :maxdepth: 1
   :caption: Legacy heartbeat examples

   examples/basic_example
   examples/train_ddp_heartbeats
