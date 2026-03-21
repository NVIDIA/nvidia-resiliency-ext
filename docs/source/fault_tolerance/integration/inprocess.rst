FT Launcher & Inprocess integration
***********************************
**FT launcher** integrates with **Inprocess recovery mechanisms**, improving fault tolerance by coordinating injob and inprocess fault recovery.

1. Heartbeat Mechanism
======================
* The **FT launcher heartbeat** remains active throughout execution to detect and mitigate potential hangs.
* Users must configure timeouts manually, ensuring they exceed **inprocess operational timeouts** to prevent conflicts.

2. Restart Policy
=================
The ``--ft-restart-policy`` argument is **deprecated**. Only ``any-failed`` is supported: the launcher restarts all workers when any worker group fails. Use of ``--ft-restart-policy`` may be removed in a future release.

Support for combining injob (FT launcher) and inprocess recovery in a single workload is being re-evaluated; a revised integration model may be documented in a future release.
