PyTorch Lightning Integration
*****************************

This section describes Fault Tolerance integration with a PTL-based workload (i.e., NeMo) using ``FaultToleranceCallback``.

1. Use ``ft_launcher`` to start the workload
============================================

Fault tolerance relies on a special launcher (``ft_launcher``), which is a modified ``torchrun``. 
If you are using NeMo, the `NeMo-Framework-Launcher <https://github.com/NVIDIA/NeMo-Framework-Launcher>`_ can be used to generate SLURM batch scripts with FT support.

2. Add the FT callback to the PTL trainer
==========================================

Add the FT callback to the PTL callbacks.

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

FT callback initialization parameters are described in the ``FaultToleranceCallback`` constructor docstring:  
:class:`nvidia_resiliency_ext.ptl_resiliency.fault_tolerance_callback.FaultToleranceCallback`

3. Implementing auto-resume
===========================

Auto-resume simplifies running training jobs that consist of multiple sequential runs.

.. note::
   Auto-resume is not part of the FT package. It is entirely implemented in a launcher script and the ``FaultToleranceCallback``.

``FaultToleranceCallback`` exposes an "interface" that allows implementing an auto-resume launcher script. Specifically, if ``autoresume=True``, 
the FT callback creates a special marker file when training is completed. The marker file location is expected to be set in the ``FAULT_TOL_FINISHED_FLAG_FILE`` environment variable.

The following steps can be used to implement an auto-resume launcher script:
   * The launcher script starts ranks with ``ft_launcher``.
   * ``FAULT_TOL_FINISHED_FLAG_FILE`` should be passed to rank processes.
   * When ``ft_launcher`` exits, the launcher script checks if the ``FAULT_TOL_FINISHED_FLAG_FILE`` file was created.

      * If ``FAULT_TOL_FINISHED_FLAG_FILE`` exists, the auto-resume loop stops, as training is complete.
      * If ``FAULT_TOL_FINISHED_FLAG_FILE`` does not exist, the continuation job can be issued (other conditions can be checked, e.g., if the maximum number of failures is not reached).