Health Check
===============================================================================

.. autoclass:: nvidia_resiliency_ext.inprocess.health_check.HealthCheck
    :special-members: __call__

.. automodule:: nvidia_resiliency_ext.inprocess.health_check
    :members:
    :exclude-members: HealthCheck

Enhanced Health Check Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The InProcess wrapper automatically includes three types of health checks when ``LOCAL_RANK`` is available:

**ChainedGPUHealthCheck**: Monitors GPU device health and recovery actions.
**ChainedNVLHealthCheck**: Monitors NVLink connectivity and link health.
**ChainedNicHealthCheck**: Monitors network interface card connectivity and link down events.

All chained health checks automatically use the ``device_index`` from ``LOCAL_RANK`` and are ready to use
immediately after construction. The underlying health check classes handle device assignment and baseline
initialization automatically.

.. note::
   The ``NicHealthCheck`` constructor now accepts an optional ``device_index`` parameter that automatically
   sets the NIC device and initializes the baseline link down counter during construction. This eliminates
   the need for manual setup and ensures accurate health monitoring from the first check.
