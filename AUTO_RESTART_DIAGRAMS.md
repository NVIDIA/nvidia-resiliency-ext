# Auto-Restart Feature Diagrams

This document contains additional diagrams to complement the main design document, providing visual representations of the system architecture, flow, and interactions.

## 1. System Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           Auto-Restart System                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │   Parent Process│    │  Child Process  │    │   External Services     │ │
│  │   (Monitor)     │    │  (Training)     │    │                         | |
│  │                 │    │                 │    │                         | |
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────────────┐ │ │
│  │ │fork_and_    │ │    │ │InProcess    │ │    │ │   TCPStore Service  │ │ │
│  │ │monitor()    │ │    │ │Wrapper      │ │    │ │                     │ │ │
│  │ │             │ │    │ │             │ │    │ │ • Hosted by Rank 0  │ │ │
│  │ │• Fork child │ │    │ │• Training   │ │    │ │ • Persists across   │ │ │
│  │ │• Monitor    │ │    │ │• Resilience │ │    │ │   restarts          │ │ │
│  │ │• Restart    │ │    │ │• Restart    │ │    │ │ • Barrier coord.    │ │ │
│  │ │  on failure │ │    │ │  logic      │ │    │ │ • State persistence │ │ │
│  │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────────────┘ │ |
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘ │
│                                                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        State Management                               │ │
│  │                                                                       │ │
│  │  • global_iteration_counter (increments by 100)                       │ │
│  │  • job_restart_counter (tracks total restarts)                        │ │
│  │  • ranks_restart_counter (per-iteration tracking)                     │ │
│  │  • State persistence across process boundaries                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────-─┘
```

## 2. Process Lifecycle Flow

```
┌─────────────┐
│ Application │
│   Start     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│fork_and_    │
│monitor()    │
│Called       │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   os.fork() │
│   Called    │
└──────┬──────┘
       │
       ├─────────────┐
       │             │
       ▼             ▼
┌─────────────┐ ┌─────────────┐
│   Parent    │ │    Child    │
│  Process    │ │   Process   │
└──────┬──────┘ └──────┬──────┘
       │               │
       │               ▼
       │         ┌─────────────┐
       │         │   Training  │
       │         │   Starts    │
       │         └──────┬──────┘
       │                │
       │                ▼
       │         ┌─────────────┐
       │         │InProcess    │
       │         │Wrapper      │
       │         └──────┬──────┘
       │                │
       │                ▼
       │         ┌─────────────┐
       │         │   Training  │
       │         │   Continues │
       │         └──────┬──────┘
       │                │
       │                ▼
       │         ┌─────────────┐
       │         │   Failure   │
       │         │   Occurs    │
       │         └──────┬──────┘
       │                │
       │                ▼
       │         ┌─────────────┐
       │         │InProcess    │
       │         │Handles      │
       │         │Restart      │
       │         └──────┬──────┘
       │                │
       │                ▼
       │         ┌─────────────┐
       │         │   Restart   │
       │         │   Logic     │
       │         └──────┬──────┘
       │                │
       │                ▼
       │         ┌─────────────┐
       │         │   Exit      │
       │         │   Code      │
       │         └─────────────┘
       │
       ▼
┌─────────────┐
│   Parent    │
│  Monitors  │
│   Child     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Parent    │
│  Checks     │
│  Exit Code  │
└──────┬──────┘
       │
       ├─Exit 0 or 130─┐
       │                │
       |                ▼
       |        ┌─────────────┐
       |        │   Parent    │
       |        │  Exits      │
       |        │  (No        │
       |        │  Restart)   │
       |        └─────────────┘
       │
    Other Exit Codes
       │            
       ▼            
┌─────────────┐ 
│   Parent    │ 
│  Detects    │ 
│  Failure    │ 
└──────┬──────┘ 
       │        
       │        
       ▼        
┌─────────────┐ 
│   Parent    │ 
│  Restarts   │
│  Child      │
└─────────────┘
       │
       ▼
┌─────────────┐
│   New       │
│   Child     │
│   Process   │
└─────────────┘
```

## 3. State Transition Diagram

```
┌────────────────-─┐
│   Initial State  │
│                  │
│ • iteration=0    │
│ • job_restart=0  │
└─────────┬──────-─┘
          │
          ▼
┌────────────────-─┐
│   Training       │
│   Running        │
│                  │
│ • iteration=0    │
│ • job_restart=0  │
└─────────┬──────-─┘
          │
          ▼
┌─────────────────┐
│   Failure       │
│   Detected      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   InProcess     │
│   Restart       │
│   Logic         │
└─────────┬───────┘
          │
          ├─────────────────┐
          │                 │
          ▼                 ▼
┌─────────────────┐ ┌─────────────────┐
│   In-Process    │ │  Across-Process │
│   Restart       │ │   Restart       │
│                 │ │                 │
│ • iteration=0   │ │ • iteration=100 │
│ • job_restart=1 │ │ • job_restart=1 │
└─────────┬───────┘ └─────────┬───────┘
          │                   │
          │                   ▼
          │         ┌─────────────────┐
          │         │   New Process   │
          │         │   Started       │
          │         │                 │
          │         │ • iteration=100 │
          │         │ • job_restart=1 │
          │         └─────────┬───────┘
          │                   │
          │                   ▼
┌─────────────────┐ ┌─────────────────┐
│   Training      │ │   Training      │
│   Continues     │ │   Continues     │
│                 │ │                 │
│ • iteration=1   │ │ • iteration=100 │
│ • job_restart=1 │ │ • job_restart=1 │
└─────────┬───────┘ └─────────┬───────┘
          │                   │
          │                   │
          ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│   InProcess     │ │   InProcess     │
│   Restart       │ │   Restart       │
│   Logic         │ │   Logic         │
└─────────┬───────┘ └─────────┬───────┘
          │                   │
          │                   │
          ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│   Training      │ │   Training      │
│   Continues     │ │   Continues     │
│                 │ │                 │
│ • iteration=2   │ │ • iteration=101 │
│ • job_restart=2 │ │ • job_restart=2 │
└─────────┬───────┘ └─────────┬───────┘
          │                   │
          │                   │
          ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│   InProcess     │ │   InProcess     │
│   Restart       │ │   Restart       │
│   Logic         │ │   Logic         │
└─────────┬───────┘ └─────────┬───────┘
          │                   │
          │                   │
          ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│   Training      │ │   Training      │
│   Continues     │ │   Continues     │
│                 │ │                 │
│ • iteration=3   │ │ • iteration=200 │
│ • job_restart=3 │ │ • job_restart=3 │
└─────────┬───────┘ └─────────┬───────┘
          │                   │
          │                   │
          ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│   Success or    │ │   Success or    │
│   Max Iterations│ │   Max Iterations│
│   Reached       │ │   Reached       │
└─────────┬───────┘ └─────────┬───────┘
          │                   │
          │                   │
          ▼                   ▼
┌──────────────-───┐ ┌─────────────-────┐
│ 0 or RestartAbort│ │ 0 or RestartAbort│
│   (Exit 130)     │ │   (Exit 130)     │
└─────────┬──────-─┘ └─────────┬─────-──┘
          │                    │
          │                    │
          ▼                    ▼
┌─────────────────┐ ┌─────────────────┐
│   Clean Exit    │ │   Clean Exit    │
│   (No Restart)  │ │   (No Restart)  │
└─────────────────┘ └─────────────────┘        
```

## 4. TCPStore Integration Flow

```
┌─────────────────┐
│   Rank 0        │
│   Process       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   TCPStore      │
│   Service       │
│   Started       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   TCPStore      │
│   Listening     │
│   on Port       │
│   MASTER_PORT+2 │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   All Ranks     │
│   Connect to    │
│   TCPStore      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Training      │
│   Begins        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Failure       │
│   Occurs        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   InProcess     │
│   Handles       │
│   Restart       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   New Process   │
│   Started       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   New Process   │
│   Connects to   │
│   Same TCPStore │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   State         │
│   Restored      │
│   from Store    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Training      │
│   Continues     │
└─────────────────┘
```

## 5. Key Space Management

```
┌────────────────────────────────────────────────────────────────-─┐
│                    Key Space Management                          │
├─────────────────────────────────────────────────────────────────-┤
│                                                                  │
│  Iteration 0:                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • global_iteration_counter = 0                              │ │
│  │ • job_restart_counter = 0                                   │ │
│  │ • Keys: iteration_0_*, etc                                  │ │
│  │ • Training session 1                                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  After Cross-Process Restart:                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • global_iteration_counter = 100                            │ │
│  │ • job_restart_counter = 1                                   │ │
│  │ • Keys: iteration_100_*, etc.                               │ │
│  │ • Training session 2                                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  After Another Cross-Process Restart:                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • global_iteration_counter = 200                            │ │
│  │ • job_restart_counter = 2                                   │ │
│  │ • Keys: iteration_200_*, etc.                               │ │
│  │ • Training session 3                                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Benefits:                                                       │
│  • No key conflicts between sessions                           │ │
│  • Clean separation of state                                   │ │
│  • Easy debugging and monitoring                               │ │
│  • Predictable key patterns                                    │ │
└──────────────────────────────────────────────────────-───────────┘
```

## 6. Exit Code Handling

```
┌─────────────────┐
│   InProcess     │
│   Wrapper       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Exception     │
│   Occurs        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Exception     │
│   Type Check    │
└─────────┬───────┘
          │
          ├─RestartAbort─┐
          │              │
          ▼              ▼
┌─────────────────┐ ┌─────────────────┐
│   sys.exit(130) │ │   Re-raise      │
│   Called        │ │   Exception     │
└─────────┬───────┘ └─────────┬───────┘
          │                   │
          ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│   Parent        │ │   Parent        │
│   Detects       │ │   Detects       │
│   Exit 130      │ │   Non-130 Exit  │
└─────────┬───────┘ └─────────┬───────┘
          │                   │
          ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│   Clean Exit    │ │   Restart       │
│   (No Restart)  │ │   Child         │
└─────────────────┘ └─────────────────┘
```


These diagrams provide a comprehensive visual understanding of the Auto-Restart feature's architecture, flow, and interactions. They complement the main design document by showing the system from different perspectives and highlighting key relationships between components.
