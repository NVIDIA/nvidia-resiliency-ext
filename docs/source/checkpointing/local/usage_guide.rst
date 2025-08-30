Usage guide
===============================================================================
The :py:class:`nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager.LocalCheckpointManager`
serves as the primary interface for leveraging local checkpointing functionality, in combination with
the :py:class:`nvidia_resiliency_ext.checkpointing.local.basic_state_dict.BasicTensorAwareStateDict`.

`LocalCheckpointManager` manages the local checkpointing path and defines replication strategies
to enhance resiliency in the event of node failures. Meanwhile,
`BasicTensorAwareStateDict` enhances the user-provided state_dict by enabling tensor-aware management,
which is essential for efficient data exchange operations critical to local checkpointing with replication.

This guide outlines the requirements, features, restrictions, and integration details for local checkpointing.

Requirements
------------

Requirements for `LocalCheckpointManager`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- The directory specified by the `root_local_ckpt_dir` parameter must have enough storage capacity to hold
  at least two checkpoint parts (clean and dirty) per rank
  multiplied by the replication factor defined by the replication strategy (`repl_strategy`).
- If a local checkpoint had been created with replication being enabled, it's recommended to enable replication also
  when loading that checkpoint, in which case the replication parameters
  (i.e. `world_size`, `--replication-jump` and `--replication-factor`) must be the same as during save.
  If replication is disabled during load, the replicas are ignored even if available which might lead to
  inability to recover from an otherwise complete checkpoint.
- All training ranks must call `LocalCheckpointManager` methods (`save`, `load`, `find_latest`) at once,
  otherwise the training ends up in a corrupted state (a NCCL collective hang or tensor allocation OOM).

Requirements for `BasicTensorAwareStateDict`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- All tensors within the user-provided state_dict must be:
    - Easily accessible (nested only within dictionaries or lists).
    - CUDA tensors (i.e., moved to GPU).
- If these requirements are not met, a custom implementation of `TensorAwareStateDict` is necessary.
    - For instance, solutions based on NVIDIA Megatron-Core should use `MCoreTensorAwareStateDict` instead.

Restrictions
------------
- `AsyncCallsQueue` must be initialized with `persistence=False`, because some local checkpointing routines
  are not pickleable. This restriction may be lifted in the future.

Functionality Overview
----------------------

Integration Overview
~~~~~~~~~~~~~~~~~~~~
Below is a simplified pseudocode example illustrating how
to integrate local checkpointing into training scripts.
This is a basic reference and may omit specific implementation details:

.. code-block:: python

    from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import LocalCheckpointManager
    from nvidia_resiliency_ext.checkpointing.local.basic_state_dict import BasicTensorAwareStateDict

    # Initialize CheckpointManager with the checkpoint directory
    ckpt_manager = LocalCheckpointManager(ckpt_dir)

    # Load the latest checkpoint if available
    iteration = ckpt_manager.find_latest()
    if iteration != -1:
        ta_state_dict, ckpt_part_id = ckpt_manager.load()
        # Use the loaded state_dict to resume training
        model.load_state_dict(ta_state_dict.state_dict)
    else:
        # An iteration value of -1 indicates that no local checkpoint was found.
        # In this case, either return an error or initialize the model from scratch.
        print('Starting training from scratch')

    # Training loop
    while True:
        # Perform a training iteration

        # Save checkpoint if conditions are met
        if save_condition():
            ta_state_dict = BasicTensorAwareStateDict(state_dict)
            ckpt_manager.save(ta_state_dict, iteration, is_async=False)

Checkpoint Replication
~~~~~~~~~~~~~~~~~~~~~~
The `LocalCheckpointManager` supports both checkpoint saving with and without replication.
To enable replication, the user must provide a `repl_strategy` argument when
constructing the `LocalCheckpointManager`.

We provide the `CliqueReplicationStrategy`, which groups ranks into rank "cliques" where
checkpoint parts are replicated.
The `replication_factor` parameter defines the size of each group (clique),
while the `replication_jump` parameter determines how the cliques are formed. Specifically:

- If `replication_jump = 1`, consecutive ranks will form a clique (e.g., ranks 0, 1, 2, etc.).
- If `replication_jump > 1`, ranks will be spaced further apart,
  with the value of replication_jump determining the gap between ranks in each clique
  (e.g., a jump of 2 would form cliques of ranks 0, 2, 4, etc.).

This approach enables flexible and scalable replication configurations,
providing fault tolerance and improving the resiliency of checkpointing across distributed systems.

During the loading process, replicated parts can be utilized to populate nodes that
do not have their respective parts stored.
The retrieval mechanism is seamlessly integrated into the LocalCheckpointManager.load method.

Asynchronous Checkpoint Saving
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `LocalCheckpointManager` supports both synchronous and asynchronous saving,
controlled by the `is_async` parameter in the `save(...)` method.

- Synchronous Save: When `is_async` is set to `False`, the `save(...)` method
  performs a blocking save operation, ensuring all data is written before returning.
- Asynchronous Save: When `is_async` is set to `True`, the `save(...)` method
  initiates a non-blocking save and returns an `AsyncRequest` object.
  This class is compatible with the `nvidia_resiliency_ext.checkpointing.async_ckpt` module.

The returned `AsyncRequest` can then be submitted to an `AsyncCallsQueue`,
enabling advanced asynchronous processing.
The usage of `AsyncRequest` with `AsyncCallsQueue` is demonstrated in the provided example,
showcasing how to efficiently manage non-blocking saves within your workflow.

.. note::
   Per the Restrictions and the included example, `AsyncCallsQueue` must be initialized with
   `persistence=False`. This is because some local checkpointing routines are not pickleable.

Logging
~~~~~~~
The :py:class:`LocalCheckpointManager` uses Python’s logging module to generate output messages.
Users can adjust the logging level or redirect logs based on their needs.
