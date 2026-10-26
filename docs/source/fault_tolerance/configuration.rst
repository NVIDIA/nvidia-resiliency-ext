Configuration and timeouts
==========================

Use explicit launcher CLI options for the recommended deployment. For example,
Megatron's section configuration is:

.. code-block:: bash

   --ft-rank-section-timeouts setup:900,step:120,checkpointing:600 \
   --ft-rank-out-of-section-timeout 180

These are example values to qualify on the target model and system. Initialization
includes checkpoint loading, compilation, and model setup. The step timeout must
cover training and evaluation. The checkpointing timeout must accommodate save
preparation and potentially expensive async finalization; background disk I/O
is not one continuously open checkpointing section.

Defaults and precedence
-----------------------

``FaultToleranceConfig`` supplies defaults, then ``--ft-cfg-path`` supplies YAML
values, then explicit ``--ft-*`` CLI arguments override those values. The YAML
file configures FT fields, not general launcher options such as ``--nnodes`` or
``--max-restarts``. Use :class:`nvidia_resiliency_ext.fault_tolerance.config.FaultToleranceConfig`
and ``ft_launcher --help`` for the configuration fields and accepted CLI syntax.

.. list-table::
   :header-rows: 1

   * - Setting
     - Default
     - Effect
   * - ``rank_section_timeouts``
     - Empty mapping
     - No named sections are monitored
   * - ``rank_out_of_section_timeout``
     - ``None``
     - No out-of-section deadline
   * - ``initial_rank_heartbeat_timeout`` / ``rank_heartbeat_timeout``
     - ``None`` / ``None``
     - Heartbeat hang detection disabled
   * - ``workload_check_interval``
     - 5 seconds
     - Timeout polling interval, not a recovery-time guarantee
   * - ``skip_section_response``
     - ``true``
     - Section messages do not wait for acknowledgement
   * - ``max_no_progress_cycles``
     - 3
     - Stop after consecutive cycles without checkpoint progress; a missing configured marker counts
   * - ``segment``
     - ``None``
     - Simple hot-spare selection without ClusterUUID requirements

Only section names present in the mapping send messages and collect timing data.
An entry with a ``null`` timeout enables section events without a finite deadline;
an omitted name skips both. CLI timeout values accept ``null`` or ``none`` to
disable a deadline. With response skipping enabled, consult the monitor logs
for section-protocol errors; disable it while debugging integration mistakes.

Optional YAML configuration
---------------------------

The same Megatron timeout settings can be supplied as:

.. code-block:: yaml

   fault_tolerance:
     rank_section_timeouts:
       setup: 900
       step: 120
       checkpointing: 600
     rank_out_of_section_timeout: 180

Pass ``--ft-cfg-path /path/to/ft.yaml`` before the workload entrypoint. Do not use
the DDP YAML unchanged for Megatron: its section names are different. Unknown
YAML fields may be ignored; check the effective configuration and section names
in the startup logs.

``ft_state.json`` is **not** this YAML configuration file. It contains calculated
monitoring timeouts saved by a workload through ``RankMonitorClient.state_dict()``.
If a workload restores calculated timeouts, they can supersede configured values.
For a fixed-timeout run use a fresh checkpoint directory or deliberately archive
the old calculated-timeout file before launch. Preserve training checkpoints.
Megatron currently restores an existing ``ft_state.json`` even when
``--calc-ft-timeouts`` is omitted.

Warmup and unmonitored intervals
--------------------------------

Megatron opens ``setup`` before initialization and closes it before the first
training or evaluation iteration. Its ``--ft-num-warmup-iters`` (default 5)
controls when it begins opening ``step``. NVRx normally suppresses out-of-section
checking until a step section has been observed when named sections are configured
without a ``warmup`` entry. Thus a finite step timeout alone does not bound hangs
in the initial warmup window.

The small GPT example uses ``--ft-num-warmup-iters 0`` and a conservative fixed
step timeout to monitor the first step too. For production, qualify startup
latency and warmup coverage against the exact Megatron/NVRx revisions. A
``warmup`` timeout is an optional monitor behavior for workloads reporting early
step sections; adding that key does not make Megatron open step sections it skips.

Experimental calibration and legacy heartbeats
----------------------------------------------

Dynamic timeout calibration is experimental and insufficiently tested for the
recommended Megatron deployment. It requires collective participation from all
ranks, representative observations, and validation after model, hardware, or I/O
changes. Megatron's ``--calc-ft-timeouts`` cannot automatically calibrate every
phase: setup needs a persistent checkpoint load, step calculation needs enough
iterations, and checkpointing calibration is skipped with async saving.

The DDP sections example offers ``--calculate_timeouts`` only for experiments;
its default uses fixed timeouts and does not load or write ``ft_state.json``.
Heartbeat APIs remain available for compatibility but are being deprecated as
a deployment approach. Their examples are under the legacy navigation. Leaving
heartbeat timeouts unset does not provide fallback protection for warmup.
