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


Using Multi-Storage Client with Async Checkpointing
---------------------------------------------------
`nvidia_resiliency_ext` supports saving checkpoints to object stores like AWS S3, Azure Blob Storage, Google Cloud Storage, and more through the NVIDIA Multi-Storage Client (MSC) integration. 

MSC (`GitHub repository <https://github.com/NVIDIA/multi-storage-client>`_) provides a unified API for various storage backends, allowing you to write checkpoints to different storage services using the same code.

Installation
^^^^^^^^^^^^
Before using MSC integration, you need to install the Multi-Storage Client package:

.. code-block:: bash
    
    # Install the Multi-Storage Client package with boto3 support
    pip install multi-storage-client[boto3]


Configuration
^^^^^^^^^^^^^

Create a configuration file for the Multi-Storage Client and export the environment variable ``MSC_PROFILE`` to point to it:

.. code-block:: bash

    export MSC_CONFIG=/path/to/your/msc_config.yaml


.. code-block:: yaml
  :caption: Example configuration file used for AWS S3.

  profiles:
    model-checkpoints:
      storage_provider:
        type: s3
        options:
          base_path: bucket-checkpoints # Set the bucket name as the base path
      credentials_provider:
        type: S3Credentials
        options:
          access_key: ${AWS_ACCESS_KEY} # Set the AWS access key in the environment variable
          secret_key: ${AWS_SECRET_KEY} # Set the AWS secret key in the environment variable


Basic Usage
^^^^^^^^^^^

To enable MSC integration, simply pass ``use_msc=True`` when creating the ``FileSystemWriterAsync`` instance:

The MSC URL scheme is ``msc://<profile-name>/<path>``. The example below shows how to save a checkpoint to the ``model-checkpoints`` profile, the data will be stored in the ``bucket-checkpoints`` bucket in AWS S3.

.. code-block:: python

    from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
   
    # Create writer with MSC enabled
    writer = FileSystemWriterAsync(
        "msc://model-checkpoints/iteration-0010",
        use_msc=True
    )


Example: Saving and Loading Checkpoints with MSC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example demonstrates a complete workflow for saving and loading checkpoints using Multi-Storage Client integration:

.. code-block:: python

    import torch
    from torch.distributed.checkpoint import (
        DefaultLoadPlanner,
        DefaultSavePlanner,
        load,
    )

    from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue, AsyncRequest
    from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
    from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
        save_state_dict_async_finalize,
        save_state_dict_async_plan,
    )

    import multistorageclient as msc


    def async_save_checkpoint(checkpoint_path, state_dict, thread_count=2):
        """
        Save checkpoint asynchronously to MSC storage.
        """
        # Create async queue
        async_queue = AsyncCallsQueue()
        
        # Create writer with MSC enabled
        writer = FileSystemWriterAsync(checkpoint_path, thread_count=thread_count, use_msc=True)
        coordinator_rank = 0
        planner = DefaultSavePlanner()
        
        # Plan the save operation
        save_state_dict_ret = save_state_dict_async_plan(
            state_dict, writer, None, coordinator_rank, planner=planner
        )
        
        # Create async request with finalization
        save_fn, preload_fn, save_args = writer.get_save_function_and_args()
        
        def finalize_fn():
            """Finalizes async checkpointing and synchronizes processes."""
            save_state_dict_async_finalize(*save_state_dict_ret)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        async_request = AsyncRequest(save_fn, save_args, [finalize_fn], preload_fn=preload_fn)
        
        # Schedule the request and return the queue for later checking
        async_queue.schedule_async_request(async_request)
        return async_queue


    def load_checkpoint(checkpoint_path, state_dict):
        """
        Load checkpoint from MSC storage into the state_dict.
        """
        # Create reader with MSC path
        reader = msc.torch.MultiStorageFileSystemReader(checkpoint_path, thread_count=2)
        
        # Load the checkpoint into the state_dict
        load(
            state_dict=state_dict,
            storage_reader=reader,
            planner=DefaultLoadPlanner(),
        )
        return state_dict


    # Initialize your model and get state_dict
    model = YourModel()
    state_dict = model.state_dict()

    # Save checkpoint asynchronously
    checkpoint_path = "msc://model-checkpoints/iteration-0010"
    async_queue = async_save_checkpoint(checkpoint_path, state_dict)
    async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)

    # Load checkpoint synchronously
    loaded_state_dict = load_checkpoint(checkpoint_path, state_dict.copy())
