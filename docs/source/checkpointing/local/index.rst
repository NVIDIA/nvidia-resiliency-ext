Local Checkpointing
===================

The local checkpointing mechanism is implemented via the Python `LocalCheckpointManager` class,
which operates on a `TensorAwareStateDict` wrapper.
This wrapper encapsulates the operations necessary for efficient replication and data transfers.

For standard models,
the provided `BasicTensorAwareStateDict` class is typically sufficient for integration.
However, for more advanced use cases, a custom `TensorAwareStateDict` implementation may be required.

To minimize saving overheads,
integrating the asynchronous version of the `LocalCheckpointManager` method is strongly recommended.

Features:

- Local saving:
  Each node saves checkpoint parts locally, either on SSDs or RAM disks, as configured by the user.
- Synchronous and asynchronous support:
  Save checkpoints either synchronously or asynchronously, based on the application's requirements.
- Automatic cleanup:
  Handles the cleanup of broken or outdated checkpoints automatically.
- Optional replication:
  The `LocalCheckpointManager.save` method supports an optional replication mechanism
  to allow checkpoint recovery in case of node failure after a restart.
- Configurable resiliency:
  The replication factor can be adjusted for enhanced resiliency.
- Latest checkpoint detection:
  The `find_latest` method in `LocalCheckpointManager` identifies the most recent complete local checkpoint.
- Automated retrieval:
  The `LocalCheckpointManager.load` method automatically retrieves valid checkpoint parts that
  are unavailable locally.

For a comprehensive description of this functionality, including detailed
requirements, restrictions, and usage examples, please refer to the :doc:`Usage
Guide <usage_guide>` and :doc:`Examples <examples>`.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage_guide
   api
   examples
