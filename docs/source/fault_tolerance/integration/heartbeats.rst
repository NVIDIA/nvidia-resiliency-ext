Heartbeats API Integration
**************************

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

Timeouts for fault detection need to be adjusted for each workload:
   * ``initial_rank_heartbeat_timeout`` should be long enough to allow for workload initialization.
   * ``rank_heartbeat_timeout`` should be at least as long as the longest possible interval between steps.

**Importantly, heartbeats are not sent during checkpoint loading and saving**, so the time for checkpoint-related operations should be taken into account.

Fixed timeout values can be used throughout the training runs, or timeouts can be calculated based on observed heartbeat intervals.  
`null` timeout values are interpreted as infinite timeouts. In such cases, values need to be calculated to make the FT usable.

.. note::
    When --ft-initial-rank-heartbeat-timeout and --ft-rank-heartbeat-timeout are not
    provided in the command-line arguments, the launcher defaults to FT's predefined values. These are
    not null/None; currently, the defaults are 60 minutes for --ft-initial-rank-heartbeat-timeout
    and 45 minutes for --ft-rank-heartbeat-timeout.

Configuration file example:

.. literalinclude:: ../../../../examples/fault_tolerance/fault_tol_cfg_heartbeats.yaml
   :language: yaml
   :linenos:


A summary of all FT configuration items can be found in :class:`nvidia_resiliency_ext.fault_tolerance.config.FaultToleranceConfig`

3. Integration with PyTorch workload code
============================================
1. Initialize a ``RankMonitorClient`` instance on each rank with ``RankMonitorClient.init_workload_monitoring()``.  
2. *(Optional)* Restore the state of ``RankMonitorClient`` instances using ``RankMonitorClient.load_state_dict()``.  
3. Periodically send heartbeats from ranks using ``RankMonitorClient.send_heartbeat()``.
4. *(Optional)* After a sufficient range of heartbeat intervals has been observed, call ``RankMonitorClient.calculate_and_set_hb_timeouts()`` to estimate timeouts.  
5. *(Optional)* Save the ``RankMonitorClient`` instance's ``state_dict()`` to a file so that computed timeouts can be reused in the next run.
6. Shut down ``RankMonitorClient`` instances using ``RankMonitorClient.shutdown_workload_monitoring()``.

Please refer to the :doc:`../examples/train_ddp_heartbeats` for an implementation example.
