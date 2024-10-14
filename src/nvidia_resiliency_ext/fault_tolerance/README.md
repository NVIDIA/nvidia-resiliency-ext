# Fault Tolerance

Fault tolerance PoC package.

Package features:
- Workload hang detection
- Automatic calculation of timeouts used for hang detection
- Detection that rank(s) terminated due to an error
- Workload respawning in case of a failure.


## High level overview:

- Each node runs single `ft_launcher` (modified `torchrun`)
- `ft_launcher` spawns rank monitors (once)
- `ft_launcher` spawns ranks (can also respawn if `--max-restarts`>0)
- Each rank uses `RankMonitorClient` to connect to its monitor (`RankMonitorServer`)
- In case of a hang, rank monitor detects missing heartbeats from its rank and terminates it
- If any ranks disappear, `ft_launcher` detects that and terminates or restarts the workload
- `ft_launcher` instances communicate via `torchrun` "rendezvous" mechanism
- Rank monitors does not communicate with each other.

```
# On a single node.
# NOTE: each rank has its separate rank monitor.

  [Rank]----(IPC)-----[Rank Monitor]
     |                     |
     |                     |
  (re/spawns)           (spawns)
     |                     |
     |                     |
[ft_launcher]---------------

```

## Usage:

1. Initialize `fault_tolerance.RankMonitorClient` instance in each rank
2. Send heartbeats from ranks using `RankMonitorClient.send_heartbeat()`
3. Run ranks using `ft_launcher`. Command line is mostly compatible with `torchrun`.  
   Additionally, you need to provide `--fault-tol-cfg-path` which is path to the FT config.

