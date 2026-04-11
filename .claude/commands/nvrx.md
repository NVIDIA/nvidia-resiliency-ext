Show all NVRx InJob slash commands and their purpose.

## Your task

Print the following table exactly:

```
NVRx Commands
═══════════════════════════════════════════════════════════════════════════════

  /nvrx-quickstart   Try InJob on 1–2 nodes with a built-in example script.
                     No production script needed. Injects a fault so you can
                     watch a restart happen. Good starting point for newcomers.

  /nvrx-create       Transform your own sbatch script to be InJob-enabled.
                     Handles all SLURM, ft_launcher, and coordination changes.

  /nvrx-submit       Upload an InJob-enabled sbatch to a cluster and submit it.
                     Manages SSH ControlMaster and writes a state file.

  /nvrx-watch        Poll a running job, collect artifacts when it finishes.
                     Reads the state file written by /nvrx-submit.

  /nvrx-validate     Validate collected artifacts: fault detection, restart
                     timing, throughput, membership, checkpoint progress.
                     Writes a report.md with PASS/FAIL for each check.

  /nvrx-run          Full pipeline in one command:
                     /nvrx-create → /nvrx-submit → /nvrx-watch → /nvrx-validate

═══════════════════════════════════════════════════════════════════════════════
New to InJob?  Start with /nvrx-quickstart
Have your own script?  Start with /nvrx-run
```
