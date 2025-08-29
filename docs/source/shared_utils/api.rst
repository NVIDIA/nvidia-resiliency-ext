API Reference
============

This section provides detailed API documentation for the NVRx Shared Utilities, focusing on the logging system.

Log Manager
----------

.. automodule:: nvidia_resiliency_ext.shared_utils.log_manager
   :members:
   :undoc-members:
   :show-inheritance:

Log Aggregator
--------------

.. automodule:: nvidia_resiliency_ext.shared_utils.log_aggregator
   :members:
   :undoc-members:
   :show-inheritance:

Log Configuration
----------------

.. automodule:: nvidia_resiliency_ext.shared_utils.log_config
   :members:
   :undoc-members:
   :show-inheritance:

Core Classes
-----------

LogManager
~~~~~~~~~~

.. autoclass:: nvidia_resiliency_ext.shared_utils.log_manager.LogManager
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

LogConfig
~~~~~~~~~

.. autoclass:: nvidia_resiliency_ext.shared_utils.log_manager.LogConfig
   :members:
   :undoc-members:
   :show-inheritance:

NodeLogAggregator
~~~~~~~~~~~~~~~~

.. autoclass:: nvidia_resiliency_ext.shared_utils.log_aggregator.NodeLogAggregator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Core Functions
-------------

setup_logger
~~~~~~~~~~~

.. autofunction:: nvidia_resiliency_ext.shared_utils.log_manager.setup_logger

Environment Variables
--------------------

.. list-table:: Environment Variables Reference
   :widths: 25 15 60
   :header-rows: 1

   * - Variable Name
     - Type
     - Description
   * - ``NVRX_NODE_LOCAL_TMPDIR``
     - string
     - Directory path for temporary log files. When set, enables distributed logging mode.
   * - ``NVRX_LOG_DEBUG``
     - string
     - Set to "1", "true", "yes", or "on" to enable DEBUG level logging. Default: INFO level.
   * - ``NVRX_LOG_TO_STDOUT``
     - string
     - Set to "1" to log to stdout instead of stderr. Default: stderr.
   * - ``NVRX_LOG_MAX_FILE_SIZE_KB``
     - integer
     - Maximum size of temporary message files in KB before rotation. Default: 10.
   * - ``NVRX_LOG_MAX_LOG_FILES``
     - integer
     - Maximum number of log files to keep per rank. Default: 4.

Configuration Methods
--------------------

.. list-table:: LogConfig Class Methods
   :widths: 30 70
   :header-rows: 1

   * - Method
     - Description
   * - ``get_node_local_tmp_dir()``
     - Returns the configured temporary directory path or None if not set
   * - ``get_log_level()``
     - Returns the configured log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   * - ``get_log_file()``
     - Returns the configured log file name
   * - ``get_max_file_size()``
     - Returns the maximum file size in bytes for log rotation
   * - ``get_max_log_files()``
     - Returns the maximum number of log files to keep
   * - ``get_log_to_stdout_cfg()``
     - Returns whether logging should go to stdout instead of stderr
   * - ``get_process_name()``
     - Returns the process name for log identification
   * - ``get_workload_rank()``
     - Returns the workload rank from RANK environment variable
   * - ``get_workload_local_rank()``
     - Returns the workload local rank from LOCAL_RANK environment variable
   * - ``get_infra_rank()``
     - Returns the infrastructure rank from SLURM_PROCID environment variable
   * - ``get_infra_local_rank()``
     - Returns the infrastructure local rank from SLURM_LOCALID environment variable
   * - ``get_node_id()``
     - Returns the node identifier for log formatting

Log Manager Properties
---------------------

.. list-table:: LogManager Properties
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Description
   * - ``node_local_tmp_logging_enabled``
     - Boolean indicating whether node local temporary logging is enabled
   * - ``workload_rank``
     - Integer representing the workload rank
   * - ``workload_local_rank``
     - Integer representing the workload local rank
   * - ``infra_rank``
     - Integer representing the infrastructure rank
   * - ``infra_local_rank``
     - Integer representing the infrastructure local rank
   * - ``logger``
     - The configured Python logging.Logger instance

Error Handling
-------------

The logging system handles various error conditions gracefully:

- **Missing Environment Variables**: Uses sensible defaults when environment variables are not set
- **Permission Errors**: Logs warnings when unable to write to specified directories
- **Rank Detection Failures**: Continues operation with available rank information
- **File System Issues**: Automatically handles log rotation and cleanup failures

Performance Considerations
------------------------

- **File Rotation**: Automatic rotation prevents temporary directories from growing indefinitely
- **Memory Usage**: Minimal memory footprint with efficient file-based message passing
- **Network Impact**: Distributed logging reduces network traffic by aggregating logs locally
- **I/O Optimization**: Buffered writing and efficient file handling for high-throughput scenarios
