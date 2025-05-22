Async Checkpointing 
=================================

The asynchronous checkpointing feature in the NVIDIA Resiliency Extension provides core utilities to offload checkpointing routines to the background.
It leverages `torch.multiprocessing` to either fork a temporary process or spawn a persistent process for efficient, non-blocking checkpointing.

Applications can monitor asynchronous checkpoint progress in a non-blocking manner
and define a custom finalization step once all ranks complete their background checkpoint saving.

This repository includes an implementation of asynchronous checkpointing utilities for both `torch.save` and `torch.distributed.save_state_dict`.
Our modified `torch.distributed.save_state_dict` interface is integrated with an optimized backend, `FileSystemWriterAsync`, which:
• Runs in the async checkpoint process creating child parallel processes for intra-node parallelism, avoiding GIL contention.
• Minimizes metadata communication overhead by metadata caching, ensuring efficient checkpoint saving.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage_guide
   api
   examples
