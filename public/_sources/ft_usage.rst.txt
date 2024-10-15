Fault Tolerance Usage Guide
###########################

Fault Tolerance is a Python package that features:
   * Workload hang detection.
   * Automatic calculation of timeouts used for hang detection.
   * Detection of rank(s) terminated due to an error.
   * Workload respawning in case of a failure.

The ``nvidia-resiliency-ext`` package also includes the PTL callback ``FaultToleranceCallback`` that simplifies FT package integration with PyTorch Lightning-based workloads.

Fault Tolerance is included in the ``nvidia_resiliency_ext.fault_tolerance`` package.
``FaultToleranceCallback`` is included in the ``nvidia_resiliency_ext.ptl_resiliency`` package.

Terms
*****
* ``PTL`` is PyTorch Lightning.
* ``Fault Tolerance``, ``FT`` is the ``fault_tolerance`` package.
* ``FT callback``, ``FaultToleranceCallback`` is a PTL callback that integrates FT with PTL.
* ``ft_launcher`` is a launcher tool included in FT, which is based on ``torchrun``.
* ``heartbeat`` is a lightweight message sent from a rank to its rank monitor that indicates that a rank is alive.
* ``rank monitor`` is a special side-process started by ``ft_launcher`` that monitors heartbeats from its rank.
* ``timeouts`` are time intervals used by a rank monitor to detect that a rank is not alive. There are two separate timeouts: for the initial heartbeat and the subsequent heartbeats.
* ``launcher script`` is a bash script that invokes ``ft_launcher``.

FT Package Design Overview
**************************

* Each node runs a single ``ft_launcher``.
* ``ft_launcher`` spawns rank monitors (once).
* ``ft_launcher`` spawns ranks (can also respawn if ``--max-restarts`` is greater than 0).
* Each rank uses ``RankMonitorClient`` to connect to its monitor (``RankMonitorServer``).
* Each rank periodically sends heartbeats to its rank monitor (e.g. after each training and evaluation step).
* In case of a hang, the rank monitor detects missing heartbeats from its rank and terminates it.
* If any ranks disappear, ``ft_launcher`` detects that and terminates or restarts the workload.
* ``ft_launcher`` instances communicate via the ``torchrun`` "rendezvous" mechanism.
* Rank monitors do not communicate with each other.

.. code-block:: text

   # Processes structure on a single node.
   # NOTE: each rank has its separate rank monitor.

   [Rank_N]----(IPC)----[Rank Monitor_N]
      |                      |
      |                      |
   (re/spawns)            (spawns)
      |                      |
      |                      |
   [ft_launcher]-------------

FT Package Usage Overview
**************************

#. Initialize a ``RankMonitorClient`` instance in each rank.
#. Send heartbeats from ranks using ``RankMonitorClient.send_heartbeat()``.
#. Run ranks using ``ft_launcher``. The command line is mostly compatible with ``torchrun``. Additionally, you might need to provide the FT config via ``--fault-tol-cfg-path`` or CLI args (``--ft-param-...``).

FT Integration Guide
********************

This section describes Fault Tolerance integration with a PTL-based workload (i.e., NeMo) using ``FaultToleranceCallback``.

1. Use ``ft_launcher`` to start the workload
============================================

Fault tolerance relies on a special launcher (``ft_launcher``), which is a modified ``torchrun``. The FT launcher runs background processes called rank monitors. 
**You need to use ft_launcher to start your workload if you are using FT**. 
For example, the `NeMo-Framework-Launcher <https://github.com/NVIDIA/NeMo-Framework-Launcher>`_ can be used to generate SLURM batch scripts with FT support.

``ft_launcher`` is similar to ``torchrun`` but it starts a rank monitor for each started rank. ``ft_launcher`` takes the FT configuration in a YAML file (``--fault-tol-cfg-path``) or via CLI args (``--ft-param-...``). 
FT configuration items are described in the :class:`FaultToleranceConfig <nvidia_resiliency_ext.fault_tolerance.config.FaultToleranceConfig>` docstring. 

2. Add FT callback to the PTL trainer
=====================================

Add the FT callback to PTL callbacks.

.. code-block:: python

   from nvidia_resiliency_ext.ptl_resiliency import FaultToleranceCallback

   fault_tol_cb = FaultToleranceCallback(
      autoresume=True,
      calculate_timeouts=True,
      logger_name="test_logger",
      exp_dir=tmp_path,
   )

   trainer = pl.Trainer(
      ...
      callbacks=[..., fault_tol_cb],
      resume_from_checkpoint=True,
   )

Core FT callback functionality includes:
   * Establishing a connection with a rank monitor.
   * Sending heartbeats during training and evaluation steps.
   * Disconnecting from a rank monitor.

Optionally, it can also:
   * Compute timeouts that will be used instead of timeouts defined in the FT config.
   * Create a flag file when the training is completed.

FT callback initialization parameters:

.. automethod:: nvidia_resiliency_ext.ptl_resiliency.fault_tolerance_callback.FaultToleranceCallback.__init__

3. Implementing auto-resume
===========================

Auto-resume is a feature that simplifies running training consisting of multiple subsequent training jobs.

.. note::
   Auto-resume is not a part of the FT package. It is entirely implemented in a launcher script and the ``FaultToleranceCallback``.

``FaultToleranceCallback`` exposes an "interface" that allows implementing an auto-resume launcher script. Specifically, if ``autoresume=True``, 
the FT callback creates a special marker file when training is completed. The marker file location is expected to be set in the ``FAULT_TOL_FINISHED_FLAG_FILE`` environment variable.

The following mechanism can be used to implement an auto-resuming launcher script:
   * The launcher script starts ranks with ``ft_launcher``.
   * ``FAULT_TOL_FINISHED_FLAG_FILE`` should be passed to rank processes.
   * When a ``ft_launcher`` exits, the launcher script checks if the ``FAULT_TOL_FINISHED_FLAG_FILE`` file was created.

      * If ``FAULT_TOL_FINISHED_FLAG_FILE`` exists, the auto-resume loop can be broken, as the training is completed.
      * If ``FAULT_TOL_FINISHED_FLAG_FILE`` does not exist, the continuation job can be issued (other conditions can be checked, e.g., if the maximum number of failures is not reached).

4. FT configuration
===================

FT configuration is passed to ``ft_launcher`` via a YAML file or CLI args, from where it's propagated to other FT components.

Timeouts for fault detection need to be adjusted for a given workload:
   * ``initial_rank_heartbeat_timeout`` should be long enough to allow for workload initialization.
   * ``rank_heartbeat_timeout`` should be at least as long as the longest possible interval between steps.

**Importantly, heartbeats are not sent during checkpoint loading and saving**, so time for checkpointing-related operations should be taken into account.

If ``calculate_timeouts: True``, timeouts will be automatically estimated based on observed intervals. Estimated timeouts take precedence over timeouts defined in the config file. 
**Timeouts are estimated at the end of a training run when checkpoint loading and saving were observed**. Hence, in a multi-part training started from scratch, 
estimated timeouts won't be available during the initial two runs. Estimated timeouts are stored in a separate JSON file.

``max_subsequent_job_failures`` allows for the automatic continuation of training on a SLURM cluster. This feature requires the SLURM job to be scheduled with ``NeMo-Framework-Launcher`` or other compatible launcher framework. 
If ``max_subsequent_job_failures`` value is `>0`, a continuation job is prescheduled. It will continue the work until ``max_subsequent_job_failures`` subsequent jobs fail (SLURM job exit code is `!= 0`) 
or the training is completed successfully ("end of training" marker file is produced by the ``FaultToleranceCallback``, i.e., due to iterations or time limit reached).

Summary of all FT configuration items:

.. autoclass:: nvidia_resiliency_ext.fault_tolerance.config.FaultToleranceConfig
   :members:
   :exclude-members: from_args, from_kwargs, from_yaml_file, to_yaml_file