Timeout diagnostics
===================

NVRx can capture Python, native CPU, and CUDA diagnostics when a selected workload
section times out. Diagnostics are opt-in; the default capture section is ``step``.
Once enabled, all completed captures are uploaded automatically. Heartbeat and
out-of-section timeouts do not trigger captures by themselves.

Enable capture
--------------

Install the requested tools in the training container and enable diagnostics
on the launcher:

.. code-block:: bash

    ft_launcher \
        --ft-timeout-dump-enabled=true \
        --ft-timeout-dump=pyspy,gdb,cuda-gdb \
        --ft-timeout-dump-gdb-command-file=/lustre/config/gdb-timeout.txt \
        --ft-timeout-dump-cuda-gdb-command-file=/lustre/config/cuda-gdb-timeout.txt \
        --ft-nvrx-logfile /lustre/run/nvrx/12345/logs/launcher.log \
        --ft-per-cycle-applog-prefix /lustre/run/logs/training.log \
        --ft-timeout-dump-local-root /tmp \
        ... train.py

Select additional named sections with, for example::

    --ft-timeout-dump-sections=step,checkpointing

The value is a comma-separated list of exact, case-sensitive workload section
names. Surrounding whitespace is ignored; empty or duplicate names are rejected.
Megatron uses ``checkpointing``, while the NVRx sections example uses
``checkpoint``. Use the names passed to ``start_section()`` by your workload.
Other names, such as ``setup`` or application-defined sections, are supported.
There are no wildcard or timeout-type aliases. Selecting a section does not
enable diagnostics or configure its timeout: it must also have a finite timeout
in ``rank_section_timeouts`` and actually expire. Heartbeat and out-of-section
timeouts are separate detection paths, not section names in this option.

Use ``--ft-timeout-dump-enabled=false`` to disable capture, including when YAML
enables it. Selecting tools alone does not enable diagnostics. The default tool
list is ``pyspy,gdb,cuda-gdb``, in that order. There is no automatic detection or
silent skipping: when enabled, the launcher validates every requested executable
and fails before starting workers if any is missing or not executable. Each
selected debugger also requires a readable command file. Supply a smaller tool
list or install/configure the missing tools. Disabled diagnostics skip tool
and command-file validation and collector setup. Dumps require
an absolute ``--ft-nvrx-logfile=<nvrx-root>/logs/<filename>`` and always use
``<nvrx-root>/diags/``. The existing requirement for
``--ft-per-cycle-applog-prefix`` with ``--ft-nvrx-logfile`` still applies.

Deployment scripts choose ``<work-dir>/nvrx/<job-id>`` for ordinary jobs and
``<work-dir>/nvrx/<array-parent-id>`` for job arrays. Diagnostics reuse that
namespace without reconstructing Slurm IDs. A root must distinguish launch
attempts whose cycle numbers restart at zero.

Successful startup validation logs resolved paths only from node0 at DEBUG
(``NVRX_LOG_DEBUG=1``): one summary for selected tools (including the CUDA-GDB
mode) and one for debugger command files, if any. Node0 uses the launcher's
infrastructure rank with hostname/topology-derived IDs disabled; Slurm job-array
offsets still apply. If the startup rank is unknown, the summaries are omitted.
Every node still validates its own dependencies. INFO is reserved for actual
capture results and upload completion. Validation errors, runtime warnings, and
capture/upload results remain visible from every node at their existing levels.

The worker obtains its monitor PID from the existing Unix socket and calls
``PR_SET_PTRACER`` to permit the monitor and its descendants to attach under
Linux Yama mode 1. This does not override container restrictions, other security
checks, or stricter Yama modes. It replaces any previously nominated ptracer;
qualify applications that already configure a debugger or crash handler.
Permission failures are logged and do not disable fault handling. Validate
attachment inside the actual training container before production rollout.

Capture targets only the training worker PID. The ptrace authorization does not
extend to separate processes created by that worker, such as DataLoader workers
or an async checkpoint-save worker. Capturing them would require their own
authorization and explicit targeting. Selecting a checkpoint section captures
the training process, which may show where checkpoint finalization is waiting;
it does not collect stacks from the async checkpoint worker itself.

CUDA-GDB launch mode
--------------------

CUDA-GDB defaults to ``batch``, preserving the existing ``--batch``, ``-ex``,
and ``-x`` invocation. To use its normal stdin command reader instead::

    --ft-timeout-dump-cuda-gdb-mode=stdin

The equivalent YAML setting is:

.. code-block:: yaml

    fault_tolerance:
      timeout_dump_cuda_gdb_mode: stdin

CLI values override YAML. Selecting a mode does not enable diagnostics or add
CUDA-GDB to the tool list. Native GDB and py-spy are unaffected. Switch back with
``--ft-timeout-dump-cuda-gdb-mode=batch``; there is no automatic fallback.

``stdin`` is fully automated: no terminal, prompts, or monitor command exchange
is required. The startup thread writes setup, ``attach``, ``source`` of the
configured command file, ``detach``, and ``quit`` into a Linux anonymous in-memory
file (``memfd``), rewinds it, and passes it as the debugger's stdin without
``--batch``. The parent closes its descriptor after subprocess startup, including
failed or abandoned launches. The shared command file is still read only by the
debugger; its path must not contain a newline or carriage return in stdin mode.
Both modes use the same capture deadline, output cap, cancellation, and cleanup.

Use stdin mode for environments where batch attach does not expose usable CUDA
state, after qualifying it in the deployed debugger/driver/container combination.
A zero exit code in stdin mode is recorded as ``completed``, not ``success``:
command errors can be printed while the debugger continues to ``detach`` and
``quit`` and exits zero. NVRx retains merged stdout/stderr and partial output,
and records the selected CUDA-GDB mode and process exit code in the capture.
It does not parse command errors or certify GPU-state availability. Nonzero exits,
timeouts, and output limits retain their existing failure statuses. Diagnostic
failure does not add retries or extend recovery beyond the capture budget.

Debugger command files
----------------------

Users must supply a command file for each selected debugger. NVRx does not
bundle default GDB or CUDA-GDB commands. Set the paths through CLI or YAML::

    --ft-timeout-dump-gdb-command-file=/lustre/config/gdb-timeout.txt
    --ft-timeout-dump-cuda-gdb-command-file=/lustre/config/cuda-gdb-timeout.txt

GDB timeout reference
~~~~~~~~~~~~~~~~~~~~~

Use this as the default deployment reference: save the following text as
``/lustre/config/gdb-timeout.txt`` and pass its path explicitly. It is a
reference for users to copy, not an automatic runtime fallback::

    # Native CPU call stacks for all host threads, up to 32 frames each.
    thread apply all bt 32

    # Loaded shared libraries and their symbol-loading status.
    info sharedlibrary

    # Process virtual address ranges and mapped files (Linux).
    info proc mappings

These commands produce text. They do not save process memory or create a core
file. Native backtraces do not automatically reconstruct Python call stacks;
py-spy provides that view. The 32-frame limit is a starting point that users
can adjust in their deployed file.

CUDA-GDB timeout reference
~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA-GDB is NVIDIA's extension of GDB for inspecting both CPU code and CUDA
GPU execution. It is a separate executable, ``cuda-gdb``. A command file is
plain text containing the commands one would otherwise type at its debugger
prompt; the ``.txt`` extension is only a naming convention. It is not a CUDA
kernel, Python program, or binary core dump.

The following reference was validated with CUDA-GDB 13.2 on four-GPU GB300
nodes. Save it as
``/lustre/config/cuda-gdb-timeout.txt``::

    # Explicit device filters work around the unfiltered context-query issue
    # observed with CUDA-GDB 13.2 on four-GPU GB300 nodes.
    echo NVRX_CUDA_GDB_STACKS_BEGIN\n
    thread apply all bt 32
    info cuda devices
    info cuda contexts device 0
    info cuda contexts device 1
    info cuda contexts device 2
    info cuda contexts device 3
    info cuda kernels
    info cuda threads
    echo NVRX_CUDA_GDB_COMMANDS_DONE\n

Adjust the device queries to the device IDs reported by ``info cuda devices``
in your deployment; the four-device list above is specific to the tested setup.
Explicit queries worked around an unfiltered ``info cuda contexts`` issue
observed in that environment. A worker using only one GPU can legitimately
report ``No CUDA contexts.`` for the other devices. Check the query for its
active device before concluding that CUDA context capture failed.

NVRx launches CUDA-GDB against the timed-out worker PID and passes this file
with ``-x`` in batch mode or ``source`` in stdin mode. Its output goes into the
CUDA-GDB text capture. Both modes use the same command file; leave ``attach``,
``detach``, and ``quit`` out of that file because NVRx supplies them.

To use this reference through stdin, add
``--ft-timeout-dump-cuda-gdb-mode=stdin`` to the launcher options. The equivalent
manual invocation below shows the command stream NVRx supplies. Replace
``<worker-pid>`` with the target PID and use a command-file path available inside
the training container:

.. code-block:: bash

    cuda-gdb --nx -q -iex 'set debuginfod enabled off' <<'EOF'
    set confirm off
    set pagination off
    set width 0
    set height 0
    attach <worker-pid>
    source /lustre/config/cuda-gdb-timeout.txt
    detach
    quit
    EOF

This invocation omits ``--batch`` and supplies ``attach`` as a top-level stdin
command. NVRx uses an anonymous in-memory file instead of a shell here-document
and also enforces the capture deadline and output limit. This manual example
does not impose those limits.

The ``info cuda``
commands describe GPU execution state; they do not dump a backtrace for every
GPU thread. Inspecting a particular GPU thread's stack requires selecting a
CUDA execution focus and then issuing ``bt``. Available detail depends on the
kernel's symbols/debug information and the debugger/driver state.

See the `NVIDIA CUDA-GDB manual
<https://docs.nvidia.com/cuda/cuda-gdb/index.html>`_ for CUDA focus selection
and the ``info cuda`` commands. CUDA-GDB capture remains best effort on a
hung GPU and needs qualification in the deployed container.

The files are independent; only selected debuggers require a file. A py-spy-only
configuration does not need either file.

NVRx disables initialization scripts and debuginfod downloads, attaches to the
worker, disables pagination, executes the file, and detaches.
NVRx still owns the subprocess deadline, output cap, and text upload. GDB stops
processing a sourced command file on an error. Batch mode reports command
failures through its exit code; stdin mode continues with subsequent top-level
commands and can exit zero despite a command error, as described above.
Each launcher checks that the selected files exist, are regular
files, and can be read before starting workers. Relative paths are resolved
once at startup from the launcher's working directory. The monitor passes the
validated path to the debugger without resolving or reading it during capture. Files may live on shared Lustre storage;
mount the path inside the training container on every node, including replacement
nodes. Debuggers read the files at capture time, so startup validation does not
guarantee their later availability.

Command files are trusted deployment code, not a sandboxed query language.
They should emit text to the captured output and leave attach/detach and process
lifecycle to NVRx. The capture byte limit applies to captured stdout/stderr;
it cannot bound files or other side effects produced by arbitrary commands.
Changes to a file take effect on the next capture; keep it stable during a
capture or deploy updates with an atomic replacement.

Binary core generation and binary artifact collection are **future extensions,
not part of the current feature**. Do not add ``gcore`` or
``generate-core-file`` to these scripts expecting NVRx to collect the result.
The collector publishes only text captures. Future core support requires its
own artifact lifecycle and capture-budget design; no core-specific options
are provided now.

Capture and upload
------------------

Each launcher forks one diagnostic collector before threaded logging and attribution
services start. The collector closes unnecessary inherited descriptors, keeps the
existing launcher log route, and remains outside training-worker cleanup.

On a selected section timeout, the rank monitor attempts one diagnostic sequence per
worker incarnation. Each tool runs in its own
subprocess, sequentially in the configured order, with bounded output in memory.
The capture grace defaults to **30 seconds total**, including startup and cleanup.
Each tool receives at most the remaining time divided by the number of remaining
tools; unused time becomes available to later tools. A small part of each share
is reserved for killing and reaping the diagnostic child. If that child remains
alive, later tools are skipped to prevent overlapping debugger attachments.
Worker termination is requested by the overall deadline, independently of
diagnostic subprocess startup and cleanup. Startup runs on a daemon thread so a
blocked executable lookup cannot hold the monitor event loop or its shutdown.
A pending launch prevents further captures; if it returns after cancellation or
timeout, its child is killed and reaped without delaying recovery. Other launchers may terminate the target sooner,
so capture is best effort. The timeout retains the original process identity
(PID and creation time): disconnecting or reconnecting its monitor socket does
not cancel termination, and a replacement process is not targeted by the old timeout.

``pyspy`` captures Python thread stacks. ``gdb`` and ``cuda-gdb`` attach to the
live worker and execute the user-supplied command files. The files determine
which native CPU stacks or CUDA state are captured. Debuggers disable
initialization scripts and debuginfod downloads, then explicitly detach.
Attachment temporarily stops the worker; detaching resumes it, so successive
captures may observe different states. Runtime capture errors are retained
and recovery continues.

GPU inspection needs GPU device access and a compatible debugger/driver stack.
CUDA-GDB receives the worker's ``TMPDIR``, ``CUDA_VISIBLE_DEVICES``, and
``CUDA_DEVICE_ORDER`` from a bounded read of its process environment. The worker
and debugger must share a writable temporary directory. Symbols and device
line/debug information improve results; production builds need not enable
``-G``. Executable validation does not prove GPU attachment will work. Qualify
GPU capture and timeout cleanup in the deployed training container. See the
`CUDA-GDB setup guide <https://docs.nvidia.com/cuda/archive/13.0.0/cuda-gdb/index.html#setting-up-the-debugger-environment>`_.

After requesting worker termination, the monitor stages the output locally::

    /tmp/nvrx-diags/cycle<N>/
        rank<local_rank>_pid<pid>_<tool>.txt.tmp
        rank<local_rank>_pid<pid>_<tool>.txt.ready

Each file contains the host, cycle, global/local rank, PID, timeout reason,
timestamp, and capture outcome. The monitor closes the file before renaming it
to ``.ready``. Failed or truncated output is retained with its status. A
still-running diagnostic child only has a pipe, never a writable staging file.

After staging, the monitor emits one INFO summary per rank and capture sequence,
combining the attempted tools' capture statuses and the local cycle directory::

    Timeout captures: cycle=0 global_rank=246 local_rank=2 pid=1536944 pyspy=success gdb=success cuda-gdb=completed dir=/tmp/nvrx-diags/cycle0

There are no per-tool staging messages at INFO or DEBUG. Individual files retain
their full metadata and status; staging failures still produce warnings.
Upload completion messages remain separate and report the shared output path
and number of captures in each node/tool upload batch.

The collector polls this local directory every 30 seconds. The launcher supplies
the local worker count; it is not hard-coded to four. A newly discovered batch
with captures from all local ranks is published immediately. An incomplete batch
waits one additional poll to collect late arrivals, then publishes whatever is
available even if ranks are still missing. New arrivals do not extend that wait.
Distinct local ranks are counted, so two PIDs for one rank do not make a batch
complete. Tools and cycles are handled independently. Waiting does not consume
an upload attempt; inputs become fixed when the first upload is attempted.

The initial polling offset is randomized within the interval to spread writes
across nodes. There are no separate quiet-interval, maximum batch-delay,
upload-jitter, or retry-delay settings, and no new timer. Training recovery never
waits for uploading. Shutdown bypasses incomplete-batch deferral for one final
collection pass within the existing shutdown grace period.
There are no new sockets, collection requests, metadata sidecars, archives, or
compression.

Each node publishes one text file per cycle and tool::

    <nvrx-root>/diags/cycle<N>/<node_name>_<tool>.txt

The collector selects a fixed batch, writes a complete ``.txt.partial`` snapshot,
syncs it, and atomically renames it over the final file in the same directory.
It syncs the directory before deleting the included local files. A late capture
requires reading the previously published contents to build a new snapshot.
Successful uploads delete their local inputs. Late captures produce updated snapshots.

Failures and retention
----------------------

* Upload failure retains local files. Each subsequent collection poll makes
  one retry per pending batch, up to the attempt limit. Every copy attempt
  rebuilds ``.partial`` from scratch.
  If publication already succeeded, an ordinary retry only finishes syncing and
  local cleanup. New arrivals are not deleted with an earlier batch. Empty cycle
  directories are retained to avoid racing with producers opening capture files;
  successful publication removes the capture files themselves.
* Exhausted batches stay local; polling does not reset their retry budget.
  Later captures for the same cycle and tool form a new batch with their own
  retry budget; they do not include or retry the exhausted files.
  Collect retained files manually after the owning launcher/collector has stopped.
  A collector crash after publication but before cleanup may leave duplicate dumps
  on replay; exactly-once delivery is deliberately not required.
* Node-local scratch is assumed to have enough space for text dumps. Each
  rank/tool capture retains its output cap, but there is no aggregate staging
  quota or pre-capture directory scan. At the default 1 MiB cap, py-spy plus
  GDB use at most 8 MiB per cycle on a four-worker node or 16 MiB on an
  eight-worker node. Failed uploads can accumulate across cycles. Local write
  failures are logged after worker termination has been requested.
* An ownership lock prevents two collectors using the same local staging root.
  Existing captures cause diagnostics to be disabled with a warning, rather
  than uploaded into a new job or deleted. Collect them or select another local
  root. Concurrent launchers on one node need distinct roots and destinations.
* The collector is not automatically restarted. Uploads do not delay training
  restart. Shutdown wakes the collector immediately for one final collection
  pass (including pending retries within their attempt limit). It grants a
  bounded drain interval, then sends SIGKILL without
  waiting indefinitely for a stuck filesystem operation. A live old writer is
  never replaced. Node loss can lose unpublished captures.

Existing NVRx logs contain concise capture/upload outcomes and destination
paths. Each capture summary includes ``cycle``, ``global_rank``, ``local_rank``,
worker ``pid``, all attempted tools' statuses, and the local cycle ``dir``. The existing log
header supplies the hostname. Dump block headers independently include ``host``,
``cycle``, ``global_rank``, ``local_rank``, and ``pid``, so the aggregated file can
be correlated with a rank in that cycle's application log without consulting
the launcher log. Dump contents remain in the diagnostic files. A ``.partial``
left by an interrupted upload is not a published snapshot.

Configuration
-------------

These fields work in the ``fault_tolerance`` YAML section. ``timeout_dump``
corresponds to ``--ft-timeout-dump``; other fields use matching
``--ft-timeout-dump-...`` options, replacing underscores with hyphens. Explicit
CLI values override YAML. Enable diagnostics with
``--ft-timeout-dump-enabled=true`` and select tools with ``--ft-timeout-dump``
in addition to the existing log configuration.

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Field
     - Default
     - Meaning
   * - ``timeout_dump_enabled``
     - ``false``
     - Enable or disable diagnostics
   * - ``timeout_dump``
     - ``pyspy,gdb,cuda-gdb``
     - Ordered comma-separated tools; all must be executable when enabled
   * - ``timeout_dump_sections``
     - ``step``
     - Comma-separated workload section names that trigger capture on timeout
   * - ``timeout_dump_local_root``
     - ``/tmp``
     - Node-local parent of ``nvrx-diags/``
   * - ``timeout_dump_pyspy``
     - ``py-spy``
     - Executable name or path
   * - ``timeout_dump_gdb``
     - ``gdb``
     - GDB executable name or path
   * - ``timeout_dump_cuda_gdb``
     - ``cuda-gdb``
     - CUDA-GDB executable name or path
   * - ``timeout_dump_gdb_command_file``
     - ``null``
     - User-supplied command file; required when gdb is selected
   * - ``timeout_dump_cuda_gdb_command_file``
     - ``null``
     - User-supplied command file; required when cuda-gdb is selected
   * - ``timeout_dump_capture_timeout``
     - ``30`` seconds
     - Total capture grace across all tools, including cleanup
   * - ``timeout_dump_max_capture_bytes``
     - ``1048576``
     - Output cap per tool, including headers
   * - ``timeout_dump_poll_interval``
     - ``30`` seconds
     - Collection and retry interval; filesystem I/O can extend the cadence
   * - ``timeout_dump_upload_attempts``
     - ``3``
     - Total attempts per batch, including the first upload
   * - ``timeout_dump_shutdown_grace``
     - ``10`` seconds
     - Maximum launcher wait for the collector to finish its final pass
