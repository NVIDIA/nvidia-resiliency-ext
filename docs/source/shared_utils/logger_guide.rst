NVRx Logger Guide
=================

The NVRx Logger is a sophisticated logging system designed specifically for multi-node, multi-rank training workloads. It provides intelligent log aggregation, rank-aware formatting, and automatic adaptation between two logging modes:

**Regular Mode**: Logs go directly to stderr/stdout (standard Python logging)
**Node Local Temporary Mode**: Each rank writes logs to temporary files on local node storage, then local rank 0 aggregates and writes them to a per-node log file

**What "Distributed Logging" Actually Means:**
The term "distributed logging" in this context refers to the fact that logs are collected from multiple ranks/processes across multiple nodes, but the actual storage and aggregation happens locally on each node. This is different from traditional distributed logging systems that send all logs to a central location over the network. The NVRx approach keeps logging local to each node for better performance and reliability.

Key Features
-----------

* **Node Local Temporary Logging**: When enabled, each rank writes logs to temporary files on local node storage, avoiding network filesystem bottlenecks
* **Automatic Log Aggregation**: Local rank 0 acts as the node aggregator, collecting logs from all ranks on the same node and writing them to a single per-node log file
* **Environment-driven Behavior**: Automatically adapts between regular logging (stderr/stdout) and node local temporary logging based on configuration
* **Fork-safe Design**: All ranks use file-based message passing to ensure child processes can log even when they don't inherit the aggregator thread
* **Dynamic Rank Detection**: Automatically reads rank information from environment variables (RANK, LOCAL_RANK, SLURM_PROCID, SLURM_LOCALID)

Architecture
-----------

The logger operates in two modes:

**Regular Mode** (default)
    Logs go directly to stderr/stdout. This is the standard Python logging behavior.

**Node Local Temporary Mode** (when ``NVRX_NODE_LOCAL_TMPDIR`` is set)
    Each rank writes logs to temporary files on local node storage (e.g., `/tmp`, `/scratch`, local SSDs). Local rank 0 aggregates these logs and writes them to a single per-node log file. This approach avoids network filesystem bottlenecks and provides better performance for high-throughput logging scenarios.

Configuration
------------

The logger is configured through environment variables. See :doc:`config_reference` for complete configuration details.

Key configuration variable:
- ``NVRX_NODE_LOCAL_TMPDIR``: Set to enable node local temporary logging with aggregation

For advanced configuration options, environment variables, and troubleshooting, refer to the :doc:`config_reference`.

Basic Usage
----------

Setup the logger at the start of your program:

.. code-block:: python

    from nvidia_resiliency_ext.shared_utils.log_manager import setup_logger
    import logging
    
    # Setup logging
    logger = setup_logger()
    
    # Get the configured logger
    log = logging.getLogger("nvrx")
    
    # Use throughout your code
    log.info("Training started")
    log.warning("GPU memory usage high")
    log.error("Rank 0 failed")

Node Local Temporary Logging Setup
--------------------------------

For workloads that need node-local temporary logging, set the environment variable:

.. code-block:: bash

    export NVRX_NODE_LOCAL_TMPDIR=/tmp/nvrx_logs

Or in your SLURM script:

.. code-block:: bash

    #!/bin/bash
    export NVRX_NODE_LOCAL_TMPDIR=/tmp/nvrx_logs
    
    srun python your_training_script.py

**⚠️ Critical Filesystem Warning**: The temporary directory experiences high write throughput from all ranks on each node. Use local node storage (e.g., `/tmp`, `/scratch`, local SSDs) and avoid network filesystems like NFS, Lustre (LFS), or any storage accessed over network.

The logger automatically handles:
- Temporary log file creation for each rank
- Log aggregation from all ranks on each node
- Per-node log file writing
- Log rotation and cleanup

Advanced Configuration
---------------------

Force logger reconfiguration for subprocesses:

.. code-block:: python

    logger = setup_logger(force_reset=True)

Log formatting automatically includes:
- Timestamp, log level, node ID
- Workload and infrastructure rank information
- Source file and line number

Example Output Format
--------------------

.. code-block:: text

    2024-01-15 10:30:45,123 [INFO] [node001] [workload:0(0) infra:0(0)] training.py:45 Training started
    2024-01-15 10:30:46,456 [WARNING] [node001] [workload:0(0) infra:0(0)] training.py:67 GPU memory usage high
    2024-01-15 10:30:47,789 [ERROR] [node001] [workload:0(0) infra:0(0)] training.py:89 Rank 0 failed

Integration with Other NVRx Components
------------------------------------

The logger automatically integrates with these NVRx components:
- **Fault Tolerance**: Automatic logging of restart events and health checks
- **In-Process Restart**: Logging of restart boundaries and process state
- **Health Check**: Logging of system health monitoring events

**Note**: Checkpointing and Straggler Detection components use their own logging mechanisms and do not integrate with the NVRx logger.

Best Practices
-------------

1. **Setup Once**: Call ``setup_logger()`` once at the start of your main program
2. **Use Standard Logger**: Access via ``logging.getLogger("nvrx")`` in other modules
3. **Environment Configuration**: Use environment variables rather than hardcoding
4. **Subprocess Handling**: Use ``force_reset=True`` for subprocesses
5. **Filesystem Selection**: Use local node storage, avoid network filesystems (NFS, Lustre)

Troubleshooting
--------------

**Common Issues:**
- **Logs not appearing**: Check ``NVRX_NODE_LOCAL_TMPDIR`` is set and directory is writable
- **Missing rank info**: Ensure RANK/LOCAL_RANK environment variables are set
- **Performance issues**: Monitor directory size, adjust file limits, verify filesystem choice (avoid NFS/Lustre)
