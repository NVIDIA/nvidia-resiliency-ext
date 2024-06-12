FT Launcher & Inprocess integration
**************************
**FT launcher** integrates with **Inprocess recovery mechanisms**, improving fault tolerance by coordinating injob and inprocess fault recovery.

1. Heartbeat Mechanism
=================
* The **FT launcher heartbeat** remains active throughout execution to detect and mitigate potential hangs.
* Users must configure timeouts manually, ensuring they exceed **inprocess operational timeouts** to prevent conflicts.

2. Worker Monitoring & Restart Policy
====================
A new ``--restart-policy`` argument in ``ft_launcher`` modifies the default worker monitor logic for better compatibility with :doc:`../inprocess/index`.

**Policy Options**

* ``min-healthy``: Restarts workers only when the number of healthy worker groups falls below <min_nodes>, as set in ``ft_launcher``.

.. note::
    For consistency, <min_nodes> should match the inprocess restarter setting.

**Behavior in min-healthy mode:**

* If enough nodes remain healthy, the worker monitor stays inactive while collaborating with :doc:`../inprocess/index`..
* If the threshold is breached, ``FT launcher`` takes over and restarts the training process.


Supported & Unsupported Configurations
====================

To ensure correct behavior with inprocess:

✅ Supported:

* ``restart-policy=min-healthy`` **(Required)**:

    * Prevents unintended upscaling.
    * Disables any-failed worker monitoring.

❌ Unsupported:

* ``any-failed`` with inprocess **(Not allowed)**:

    * Incompatible with inprocess restarts.
    * Causes FT launcher to misinterpret terminated processes as failures, triggering unnecessary restarts.
    * Enables upscaling, allowing FT launcher to restart training when a new node becomes available.
    * Can lead to undefined behavior when combined with inprocess restarts.

In short, ``any-failed`` must not be used with inprocess, as it disrupts the intended fault recovery process.

Please refer to the :doc:`../examples/in_job_and_in_process_example` for an implementation example.
