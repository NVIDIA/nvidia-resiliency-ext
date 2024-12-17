Usage guide
===============================================================================
The :py:class:`nvidia_resiliency_ext.checkpointing.async_ckpt.core.AsyncCallsQueue`
provides application users with an interface to schedule :py:class:`nvidia_resiliency_ext.checkpointing.async_ckpt.core.AsyncRequest`, 
which defines checkpoint routine, its args/kwargs and finalization steps when the checkpoint routine is finished.

:py:class:`nvidia_resiliency_ext.checkpointing.async_ckpt.torch_ckpt.TorchAsyncCheckpoint` 
           is an instatiation of the core utilities to make `torch.save` run asynchronously.


The implementation assumes all training ranks creates :py:class:`core.AsyncCallsQueue` and synchronize with :py:class:`core.AsyncCallsQueue.maybe_finalize_async_calls` by default.


Requirements
------------
:py:class:`nvidia_resiliency_ext.checkpointing.utils` includes a couple of routines used for :py:class:`nvidia_resiliency_ext.checkpointing.async_ckpt.core`
:py:class:`nvidia_resiliency_ext.checkpointing.utils.wrap_for_async` disables garbage collection in a forked process to run user's checkpoint routine
to prevent failures incurred by GC, which tries to deallocate CUDA tensors in a forked process.
This routine requires the first argument of the passed user fn should be state dictionary containing tensors or objects for checkpoint
 
The current implementation uses a forked process to run pre-staged tensors in host memory by pinned memcpy. 
So, the routine should include :py:class:`nvidia_resiliency_ext.checkpointing.utils.preload_tensors` to stage GPU tensors in a state dictionary to host memory before it's passed to `AsyncCallsQueue`


Synchronization of Asynchronous Checkpoint Requests
---------------------------------------------------
The class :py:class:`nvidia_resiliency_ext.checkpointing.async_ckpt.core.AsyncCallsQueue`
provides a method to verify whether asynchronous checkpointing has completed in the background.
Each trainer can check the status of its forked checkpoint process by calling
:py:meth:`nvidia_resiliency_ext.checkpointing.async_ckpt.core.AsyncCallsQueue.maybe_finalize_async_calls`
with `blocking=False`.

When a trainer needs to finalize all active checkpoint requests in a blocking manner, it can call the same method with `blocking=True`.

Additionally,
:py:meth:`AsyncCallsQueue.maybe_finalize_async_calls` includes another parameter that must be set to `no_dist=False` when global synchronization across all ranks is required.
For example, if a checkpointing routine needs to write metadata (e.g., iteration or sharding information) after completing a set of checkpoints,
global synchronization ensures that all ranks finish their asynchronous checkpointing before proceeding.

This global synchronization is implemented using a single integer collective operation, ensuring that all ranks have completed their asynchronous checkpoint writes.
The synchronization logic is handled within
:py:meth:`nvidia_resiliency_ext.checkpointing.async_ckpt.core.DistributedAsyncCaller.is_current_async_call_done`, which is invoked by :py:meth:`AsyncCallsQueue.maybe_finalize_async_calls`.

The following snippet demonstrates how global synchronization is performed when `no_dist` is set to `False` (indicating that synchronization is required):

.. code-block:: python

   is_alive = int(self.process.is_alive()) if self.process is not None else 0

   is_done = is_alive
   if not no_dist:
       ten = torch.tensor([is_alive], dtype=torch.int, device=torch.cuda.current_device())



`TorchAsyncCheckpoint` wraps around these synchronization routines in `nvidia_resiliency_ext.checkpointing.async_ckpt.TorchAsyncCheckpoint.finalize_async_save`.
The following example shows how the routine can be used to synchronize the asynchronous checkpoint in a non-blocking / blocking manner

.. code-block:: python

    from nvidia_resiliency_ext.checkpointing.async_ckpt import TorchAsyncCheckpoint
    ...
    async_impl = TorchAsyncCheckpoint

    # Training loop
    while True:
 	async_impl.finalize_async_save(blocking=False)
        # Perform a training iteration
	...
        # Save checkpoint if conditions are met
        if save_condition():
		async_impl.async_save(model.state_dict(), ckpt_dir)

    async_impl.finalize_async_save(blocking=True)
