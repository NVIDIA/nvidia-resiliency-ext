API Reference
============

This section provides detailed API documentation for the NVRx Shared Utilities, focusing on the logging system.

.. note::
   For configuration options, environment variables, examples, and usage guides, see :doc:`config_reference`.

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

Quick Reference
--------------

For quick access to configuration options and environment variables, see the :doc:`config_reference` page which contains:

- Complete environment variables reference
- Configuration examples and integration guides
- Best practices and troubleshooting
- Performance considerations and filesystem selection
