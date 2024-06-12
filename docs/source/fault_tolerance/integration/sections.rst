Sections API Integration
************************

1. Prerequisites
=================
* Run ranks using ``ft_launcher``. The command line is mostly compatible with ``torchrun``. 
* Pass the FT config to the ``ft_launcher``.

.. note::
   Some clusters (e.g., SLURM) use SIGTERM as a default method of requesting a graceful workload shutdown.
   It is recommended to implement appropriate signal handling in a fault-tolerant workload.
   To avoid deadlocks and other unintended side effects, signal handling should be synchronized across all ranks.

2. FT configuration
====================

With the section-based API, timeouts must be set for all defined sections, which wrap operations like
training/eval steps, checkpoint saving, and initialization. Additionally, an out-of-section timeout
applies when no section is active.

.. note::
    Ensure out-of-section timeout is long enough to accommodate restart overhead, as excessively small values can cause imbalance.
    If needed, consider merging sections (e.g., moving 'init' into 'out-of-section') to provide more buffer time.

Relevant FT configuration items are:
   * ``rank_section_timeouts`` is a map of a section name to its timeout.
   * ``rank_out_of_section_timeout`` is the out-of-section timeout.

Fixed timeout values can be used throughout the training runs, or timeouts can be calculated based on observed intervals.  
`null` timeout values are interpreted as infinite timeouts. In such cases, values need to be calculated to make the FT usable.

Config file example:

.. literalinclude:: ../../../../examples/fault_tolerance/fault_tol_cfg_sections.yaml
   :language: yaml
   :linenos:

A summary of all FT configuration items can be found in :class:`nvidia_resiliency_ext.fault_tolerance.config.FaultToleranceConfig`

3. Integration with PyTorch workload code
============================================
1. Initialize a ``RankMonitorClient`` instance on each rank with ``RankMonitorClient.init_workload_monitoring()``.  
2. *(Optional)* Restore the state of ``RankMonitorClient`` instances using ``RankMonitorClient.load_state_dict()``.  
3. Mark some sections of the code with ``RankMonitorClient.start_section('<section name>')`` and ``RankMonitorClient.end_section('<section name>')``.
4. *(Optional)* After a sufficient range of section intervals has been observed, call ``RankMonitorClient.calculate_and_set_section_timeouts()`` to estimate timeouts.  
5. *(Optional)* Save the ``RankMonitorClient`` instance's ``state_dict()`` to a file so that computed timeouts can be reused in the next run.
6. Shut down ``RankMonitorClient`` instances using ``RankMonitorClient.shutdown_workload_monitoring()``.

Please refer to the :doc:`../examples/train_ddp_sections` for an implementation example.
