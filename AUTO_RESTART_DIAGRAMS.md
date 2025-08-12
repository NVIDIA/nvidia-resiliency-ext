# Auto-Restart Feature Diagrams

This document contains additional diagrams to complement the main design document, providing visual representations of the system architecture, flow, and interactions.

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Auto-Restart System                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │   Parent Process│    │  Child Process  │    │   External Services     │ │
│  │   (Monitor)     │    │  (Training)     │ │
│  │                 │    │                 │ │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────────────┐ │ │
│  │ │fork_and_    │ │    │ │InProcess    │ │    │ │   TCPStore Service  │ │ │
│  │ │monitor()    │ │    │ │Wrapper      │ │    │ │                     │ │ │
│  │ │             │ │    │ │             │ │    │ │ • Hosted by Rank 0  │ │ │
│  │ │• Fork child │ │    │ │• Training   │ │    │ │ • Persists across   │ │ │
│  │ │• Monitor    │ │    │ │• Resilience │ │    │ │   restarts          │ │ │
│  │ │• Restart    │ │    │ │• Restart    │ │    │ │ • Barrier coord.    │ │ │
│  │ │  on failure│ │    │ │  logic      │ │    │ │ • State persistence │ │ │
│  │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────────────┘ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        State Management                                │ │
│  │                                                                         │ │
│  │  • global_iteration_counter (increments by 100)                        │ │
│  │  • job_restart_counter (tracks total restarts)                         │ │
│  │  • ranks_restart_counter (per-iteration tracking)                      │ │
│  │  • State persistence across process boundaries                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
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
       │                │
       │                ▼
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
       │         └──────┬──────┘
       │                │
       │                ▼
       │         ┌─────────────┐
       │         │Exit Code    │
       │         │130?         │
       │         └──────┬──────┘
       │                │
       │                ├─Yes──┐
       │                │      │
       │                ▼      ▼
       │         ┌─────────────┐ ┌─────────────┐
       │         │   Clean     │ │   Parent   │
       │         │   Exit      │ │  Exits     │
       │         │   (No       │ │  (No       │
       │         │   Restart)  │ │  Restart)  │
       │         └─────────────┘ └─────────────┘
       │
       │                ├─No───┐
       │                │      │
       │                ▼      ▼
       │         ┌─────────────┐ ┌─────────────┐
       │         │   Parent    │ │   Parent   │
       │         │   Detects   │ │  Waits     │
       │         │   Failure   │ │  &         │
       │         └──────┬──────┘ │  Restarts  │
       │                │        └──────┬─────┘
       │                ▼               │
       │         ┌─────────────┐       │
       │         │   Parent    │       │
       │         │   Restarts  │◄──────┘
       │         │   Child     │
       │         └─────────────┘
       │
       ▼
┌─────────────┐
│   Parent    │
│  Exits     │
└─────────────┘
```

## 3. State Transition Diagram

```
┌─────────────────┐
│   Initial State │
│                 │
│ • iteration=0   │
│ • job_restart=0 │
│ • ranks_restart=0│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Training      │
│   Running       │
│                 │
│ • iteration=0   │
│ • job_restart=0 │
│ • ranks_restart=0│
└─────────┬───────┘
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
│ • ranks_restart=1│ │ • ranks_restart=0│
└─────────┬───────┘ └─────────┬───────┘
          │                   │
          │                   ▼
          │         ┌─────────────────┐
          │         │   New Process   │
          │         │   Started       │
          │         │                 │
          │         │ • iteration=100 │
          │         │ • job_restart=1 │
          │         │ • ranks_restart=0│
          │         └─────────┬───────┘
          │                   │
          │                   ▼
          │         ┌─────────────────┐
          │         │   Training      │
          │         │   Continues     │
          │         │                 │
          │         │ • iteration=100 │
          │         │ • job_restart=1 │
          │         │ • ranks_restart=0│
          │         └─────────┬───────┘
          │                   │
          │                   ▼
          │         ┌─────────────────┐
          │         │   Success or    │
          │         │   Max Iterations│
          │         │   Reached       │
          │         └─────────┬───────┘
          │                   │
          │                   ▼
          │         ┌─────────────────┐
          │         │   RestartAbort  │
          │         │   (Exit 130)    │
          │         └─────────┬───────┘
          │                   │
          │                   ▼
          │         ┌─────────────────┐
          │         │   Clean Exit    │
          │         │   (No Restart)  │
          │         └─────────────────┘
          │
          ▼
┌─────────────────┐
│   Training      │
│   Continues     │
│                 │
│ • iteration=0   │
│ • job_restart=1 │
│ • ranks_restart=1│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Next          │
│   Iteration     │
│   or Success    │
└─────────────────┘
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
┌─────────────────────────────────────────────────────────────────┐
│                    Key Space Management                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Iteration 0:                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • global_iteration_counter = 0                             │ │
│  │ • Keys: iteration_0_*, job_restart_0_*, ranks_restart_0_*  │ │
│  │ • Training session 1                                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  After Cross-Process Restart:                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • global_iteration_counter = 100                           │ │
│  │ • Keys: iteration_100_*, job_restart_100_*, etc.           │ │
│  │ • Training session 2                                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  After Another Cross-Process Restart:                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • global_iteration_counter = 200                           │ │
│  │ • Keys: iteration_200_*, job_restart_200_*, etc.           │ │
│  │ • Training session 3                                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Benefits:                                                      │
│  • No key conflicts between sessions                           │ │
│  • Clean separation of state                                   │ │
│  • Easy debugging and monitoring                               │ │
│  • Predictable key patterns                                    │ │
└─────────────────────────────────────────────────────────────────┘
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

## 7. Hot Spare Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hot Spare Configuration                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Total World Size: 8                                            │
│  Active World Size: 6                                           │
│  Hot Spare Count: 2                                             │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Rank 0        │ │   Rank 1        │ │   Rank 2        │   │
│  │   (Active)      │ │   (Active)      │ │   (Active)      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Rank 3        │ │   Rank 4        │ │   Rank 5        │   │
│  │   (Active)      │ │   (Active)      │ │   (Active)      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐                       │
│  │   Rank 6        │ │   Rank 7        │                       │
│  │   (Hot Spare)   │ │   (Hot Spare)   │                       │
│  └─────────────────┘ └─────────────────┘                       │
│                                                                 │
│  Benefits:                                                      │
│  • Fault tolerance for rank failures                           │
│  • Reduced restart overhead                                    │
│  • Better resource utilization                                 │
│  • Configurable resilience level                               │
└─────────────────────────────────────────────────────────────────┘
```

## 8. Component Interaction Sequence

```
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│Training │ │InProcess│ │TCPStore │ │Parent   │ │State    │
│App      │ │Wrapper  │ │Service  │ │Process  │ │Store    │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │           │           │           │           │
     │───Start──▶│           │           │           │
     │           │           │           │           │
     │           │───Init───▶│           │           │
     │           │           │           │           │
     │           │◀──Ready───│           │           │
     │           │           │           │           │
     │           │───State──▶│           │           │
     │           │           │           │           │
     │           │◀──State───│           │           │
     │           │           │           │           │
     │           │───Train──▶│           │           │
     │           │           │           │           │
     │           │◀──Failure─│           │           │
     │           │           │           │           │
     │           │───Restart▶│           │           │
     │           │           │           │           │
     │           │───State──▶│           │           │
     │           │           │           │           │
     │           │◀──State───│           │           │
     │           │           │           │           │
     │           │───Exit───▶│           │           │
     │           │           │           │           │
     │           │           │           │───Exit───▶│
     │           │           │           │           │
     │           │           │           │◀──Code───│
     │           │           │           │           │
     │           │           │           │───Check──▶│
     │           │           │           │           │
     │           │           │           │◀──Result─│
     │           │           │           │           │
     │           │           │           │───Action─▶│
     │           │           │           │           │
```

These diagrams provide a comprehensive visual understanding of the Auto-Restart feature's architecture, flow, and interactions. They complement the main design document by showing the system from different perspectives and highlighting key relationships between components.
