Async Checkpointing 
=================================

Asynchronous checkpointing in NVIDIA resiliency extension provides core utilities to make checkpointing routines
run in the background. It uses `torch.multiprocessing` to fork a temporary process to initiate asynchronous checkpointing routine.
Application can check this asynchronous checkpoint save in a non-blocking manner and specify a user-defined finalization step 
when all ranks finish its background checkpoint saving.

In this repo, we include an instantation of this asynchronous checkpoint utils for `torch.save`


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage_guide
   api
   examples
