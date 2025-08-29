NVRx Logger Guide
=================

The NVRx Logger is a sophisticated distributed logging system designed specifically for multi-node, multi-rank training workloads. It provides intelligent log aggregation, rank-aware formatting, and automatic adaptation between distributed and regular logging modes.

Key Features
-----------

* **Distributed Logging**: When enabled, each node logs independently to avoid overwhelming centralized logging systems
* **Automatic Aggregation**: Local rank 0 acts as the node aggregator, collecting logs from all ranks on the same node
* **Environment-driven Behavior**: Automatically adapts between distributed and regular logging based on configuration
* **Fork-safe Design**: All ranks use file-based message passing to ensure child processes can log even when they don't inherit the aggregator thread
* **Dynamic Rank Detection**: Automatically reads rank information from environment variables (RANK, LOCAL_RANK, SLURM_PROCID, SLURM_LOCALID)

Architecture
-----------

The logger operates in two modes:

**Regular Mode** (default)
    Logs go directly to stderr/stdout. This is the standard Python logging behavior.

**Distributed Mode** (when ``NVRX_NODE_LOCAL_TMPDIR`` is set)
    Each rank writes logs to temporary files in the specified directory. Local rank 0 aggregates these logs and writes them to a per-node log file.

Configuration
------------

The logger is configured through environment variables:

.. list-table:: Environment Variables
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Description
   * - ``NVRX_NODE_LOCAL_TMPDIR``
     - Directory for temporary log files. When set, enables distributed logging with aggregation.
   * - ``NVRX_LOG_DEBUG``
     - Set to "1", "true", "yes", or "on" to enable DEBUG level logging (default: INFO)
   * - ``NVRX_LOG_TO_STDOUT``
     - Set to "1" to log to stdout instead of stderr
   * - ``NVRX_LOG_MAX_FILE_SIZE_KB``
     - Maximum size of temporary message files in KB before rotation (default: 10)
   * - ``NVRX_LOG_MAX_LOG_FILES``
     - Maximum number of log files to keep per rank (default: 4)

Basic Usage
----------

Setup the logger at the start of your program:

.. code-block:: python

    from nvidia_resiliency_ext.shared_utils.log_manager import setup_logger
    
    # Basic setup
    logger = setup_logger()
    
    # With custom temporary directory
    logger = setup_logger(node_local_tmp_dir="/tmp/my_logs")
    
    # With custom prefix for distributed logging
    logger = setup_logger(node_local_tmp_prefix="mytraining")

Use the logger throughout your code:

.. code-block:: python

    import logging
    
    # Get the configured logger
    logger = logging.getLogger("nvrx")
    
    # Log messages
    logger.info("Training started")
    logger.warning("GPU memory usage high")
    logger.error("Rank 0 failed")

Distributed Logging Setup
------------------------

For distributed training workloads, set the environment variable:

.. code-block:: bash

    export NVRX_NODE_LOCAL_TMPDIR=/tmp/nvrx_logs

Or in your SLURM script:

.. code-block:: bash

    #!/bin/bash
    export NVRX_NODE_LOCAL_TMPDIR=/tmp/nvrx_logs
    
    srun python your_training_script.py

The logger will automatically:
1. Create temporary log files for each rank
2. Aggregate logs from all ranks on each node
3. Write aggregated logs to a per-node log file
4. Handle log rotation and cleanup

Advanced Configuration
---------------------

Force logger reconfiguration (useful for subprocesses):

.. code-block:: python

    # Force fresh logger setup
    logger = setup_logger(force_reset=True)

Custom log formatting is automatically applied, including:
- Timestamp
- Log level
- Node ID
- Workload rank and local rank
- Infrastructure rank and local rank
- Source file and line number

Example Output Format
--------------------

.. code-block:: text

    2024-01-15 10:30:45,123 [INFO] [node001] [workload:0(0) infra:0(0)] training.py:45 Training started
    2024-01-15 10:30:46,456 [WARNING] [node001] [workload:0(0) infra:0(0)] training.py:67 GPU memory usage high
    2024-01-15 10:30:47,789 [ERROR] [node001] [workload:0(0) infra:0(0)] training.py:89 Rank 0 failed

Integration with Other NVRx Components
------------------------------------

The logger is automatically used by other NVRx components:

- **Fault Tolerance**: Automatic logging of restart events and health checks
- **In-Process Restart**: Logging of restart boundaries and process state
- **Checkpointing**: Logging of checkpoint operations and progress
- **Straggler Detection**: Logging of performance metrics and detection events

Best Practices
-------------

1. **Setup Once**: Call ``setup_logger()`` once at the start of your main program
2. **Use Standard Logger**: Access the logger via ``logging.getLogger("nvrx")`` in other modules
3. **Environment Configuration**: Use environment variables for configuration rather than hardcoding
4. **Subprocess Handling**: Use ``force_reset=True`` when setting up logging in subprocesses
5. **Directory Permissions**: Ensure the temporary directory has proper write permissions for all ranks

Troubleshooting
--------------

**Logs not appearing in distributed mode:**
- Check that ``NVRX_NODE_LOCAL_TMPDIR`` is set
- Verify directory permissions
- Check that local rank 0 is running the aggregator

**Missing rank information:**
- Ensure environment variables (RANK, LOCAL_RANK) are set
- Check that the logger is configured before rank information is needed

**Performance issues:**
- Monitor temporary directory size
- Adjust ``NVRX_LOG_MAX_FILE_SIZE_KB`` and ``NVRX_LOG_MAX_LOG_FILES`` as needed
