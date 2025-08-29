Configuration Reference
======================

This is a quick reference for all NVRx Logger configuration options.

Environment Variables
--------------------

.. list-table:: Complete Environment Variables Reference
   :widths: 25 15 20 40
   :header-rows: 1

   * - Variable
     - Type
     - Default
     - Description
   * - ``NVRX_NODE_LOCAL_TMPDIR``
     - string
     - None
     - Directory for temporary log files. When set, enables distributed logging mode.
   * - ``NVRX_LOG_DEBUG``
     - string
     - INFO
     - Set to "1", "true", "yes", or "on" to enable DEBUG level logging.
   * - ``NVRX_LOG_TO_STDOUT``
     - string
     - stderr
     - Set to "1" to log to stdout instead of stderr.
   * - ``NVRX_LOG_MAX_FILE_SIZE_KB``
     - integer
     - 10
     - Maximum size of temporary message files in KB before rotation.
   * - ``NVRX_LOG_MAX_LOG_FILES``
     - integer
     - 4
     - Maximum number of log files to keep per rank.

Python API Parameters
--------------------

.. list-table:: setup_logger Function Parameters
   :widths: 30 15 20 35
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``node_local_tmp_dir``
     - string
     - None
     - Custom temporary directory path. Overrides NVRX_NODE_LOCAL_TMPDIR.
   * - ``force_reset``
     - boolean
     - False
     - Force reconfiguration even if logger is already configured.
   * - ``node_local_tmp_prefix``
     - string
     - None
     - Custom prefix for log files in distributed mode.

LogManager Constructor Parameters
-------------------------------

.. list-table:: LogManager Class Parameters
   :widths: 30 15 20 35
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``node_local_tmp_dir``
     - string
     - None
     - Directory path for temporary log files.
   * - ``node_local_tmp_prefix``
     - string
     - None
     - Prefix for log files in distributed mode.

Configuration Examples
---------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Enable distributed logging
    export NVRX_NODE_LOCAL_TMPDIR=/tmp/nvrx_logs
    
    # Enable debug logging
    export NVRX_LOG_DEBUG=1

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Complete configuration
    export NVRX_NODE_LOCAL_TMPDIR=/tmp/nvrx_logs
    export NVRX_LOG_DEBUG=1
    export NVRX_LOG_TO_STDOUT=1
    export NVRX_LOG_MAX_FILE_SIZE_KB=100
    export NVRX_LOG_MAX_LOG_FILES=10

Python Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nvidia_resiliency_ext.shared_utils.log_manager import setup_logger
    
    # Custom configuration
    logger = setup_logger(
        node_local_tmp_dir="/custom/logs",
        node_local_tmp_prefix="mytraining",
        force_reset=True
    )

SLURM Integration
----------------

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=nvrx_training
    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=8
    
    # NVRx Logger Configuration
    export NVRX_NODE_LOCAL_TMPDIR=/tmp/nvrx_logs_${SLURM_JOB_ID}
    export NVRX_LOG_DEBUG=1
    export NVRX_LOG_MAX_FILE_SIZE_KB=100
    
    # Launch training
    srun python training_script.py

Docker Integration
-----------------

.. code-block:: dockerfile

    # Dockerfile
    FROM nvcr.io/nvidia/pytorch:24.01-py3
    
    # Install NVRx
    RUN pip install nvidia-resiliency-ext
    
    # Set default logging configuration
    ENV NVRX_NODE_LOCAL_TMPDIR=/tmp/nvrx_logs
    ENV NVRX_LOG_DEBUG=1

Kubernetes Integration
---------------------

.. code-block:: yaml

    # kubernetes-deployment.yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: nvrx-training
    spec:
      template:
        spec:
          containers:
          - name: training
            image: nvrx-training:latest
            env:
            - name: NVRX_NODE_LOCAL_TMPDIR
              value: "/tmp/nvrx_logs"
            - name: NVRX_LOG_DEBUG
              value: "1"

Configuration Precedence
-----------------------

1. **Python API parameters** (highest priority)
2. **Environment variables**
3. **Default values** (lowest priority)

Example:
- If you set `NVRX_NODE_LOCAL_TMPDIR=/tmp/env_logs` in environment
- And call `setup_logger(node_local_tmp_dir="/tmp/api_logs")`
- The API parameter `/tmp/api_logs` will be used

Best Practices
--------------

✅ **Do:**
- Set `NVRX_NODE_LOCAL_TMPDIR` for distributed training
- Use job-specific directories (e.g., `/tmp/nvrx_logs_${SLURM_JOB_ID}`)
- Enable debug logging during development
- Use appropriate file size limits for your workload

❌ **Don't:**
- Use system-critical directories (e.g., `/var/log`)
- Set extremely large file size limits
- Keep too many log files (can fill disk)
- Mix different logging configurations in the same job

Troubleshooting
---------------

**Common Issues:**

.. list-table:: Troubleshooting Guide
   :widths: 30 70
   :header-rows: 1

   * - Issue
     - Solution
   * - Logs not appearing
     - Check `NVRX_NODE_LOCAL_TMPDIR` is set and writable
   * - Permission denied
     - Ensure directory has proper write permissions
   * - Disk space issues
     - Reduce `NVRX_LOG_MAX_FILE_SIZE_KB` or `NVRX_LOG_MAX_LOG_FILES`
   * - Missing rank info
     - Verify RANK and LOCAL_RANK environment variables are set
   * - Performance issues
     - Monitor temporary directory size and adjust limits

**Debug Mode:**
Enable debug logging to see detailed configuration information:

.. code-block:: bash

    export NVRX_LOG_DEBUG=1
    python your_script.py

This will show:
- Current configuration values
- Directory creation status
- Rank detection results
- Log handler setup details
