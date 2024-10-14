# Nvidia Resiliency Extension

This project combines multiple resiliency-related solutions.
- Fault Tolerance package
- Straggler Detection package
- PyTorch Lightning callbacks


## Installation:

### From sources
- `git clone --recursive <this repo URL>`
- `cd <repo>`
- `pip install .`

Requirements:
- Python >= 3.10
- gcc >= 8.0
- CUDA >= 11.8

## Fault Tolerance integration guide

This section describes Fault Tolerance callback integration with a PTL-based workload (e.g. NeMo).

Let's define some terms used in this section:
- `PTL` is PyTorch Lightning
- `Fault Tolerance`, `FT` is the `fault_tolerance` package, included in `nvidia_resiliency_ext`. 
- `FT callback`, `FaultToleranceCallback` is a PTL callback defined in `ptl_resiliency` package, included in `nvidia_resiliency_ext`.
- `ft_launcher` is a launcher tool included in the FT, which is based on `torchrun`.  
- `heartbeat` is a lightweight message sent from a rank to its rank monitor that indicates that a rank is alive.
- `rank monitor` is a special side-process started by `ft_launcher` that monitors heartbeats from its rank.
- `timeouts` are time intervals used by a rank monitor to detect that a rank is not alive. 
    There are 2 separate timeouts: for the initial heartbeat and the subsequent heartbeats.
- `launcher script` is a bash script that invokes `ft_launcher`.

### 0. Use `ft_launcher` to start the workload

`ft_launcher` is similar to `torchrun` but it starts a rank monitor for each started rank.  
`ft_launcher` takes the FT configuration in a YAML file (`--fault-tol-cfg-path`) or via CLI args (`--ft-param-...`).  
FT configuration items are described in `FaultToleranceConfig` docstring.

### 1. Add FT callback to the trainer

Add FT callback to PTL callbacks. 

```
fault_tol_cb = FaultToleranceCallback(
    autoresume=True,
    calculate_timeouts=True,
    logger_name="test_logger",
    exp_dir=tmp_path,
)

trainer = pl.Trainer(
    ...
    callbacks=[..., fault_tol_cb],
)
```


Core FT callback functionality is:
- Establishing a connection with a rank monitor
- Sending heartbeats during training and evaluation steps
- Disconnecting from a rank monitor

Optionally, it can also:
- Compute timeouts that will be used instead of timeouts defined in the FT config
- Create a flag file when the training is completed

FT callback initialization params:
```
def __init__(
    self,
    autoresume: bool,
    calculate_timeouts: bool,
    simulated_fault_params: Optional[Any] = None,
    exp_dir: Union[str, pathlib.Path, None] = None,
    logger_name: Optional[str] = "nemo_logger.FaultToleranceCallback",
):
    """
    Initialize callback instance.

    This is a lightweight initialization. Most of the initialization is conducted in the 'setup' hook.

    Args:
        autoresume (bool): Set to `True` if the FT auto-resume feature is used (e.g., there are multiple training jobs to be run).
        calculate_timeouts (bool): Set to `True` if FT timeouts should be calculated based on observed heartbeat intervals.
            Calculated timeouts overwrite the timeouts from the FT config.
            Timeouts are computed at the end of a training job, if there was checkpoint loading and saving.
            For example, for training started from scratch, the timeouts are computed at the end of the second job.
        simulated_fault_params (Optional[Any], optional): Simulated fault spec. It's for debugging only. Defaults to None.
        exp_dir (Union[str, pathlib.Path, None], optional): Directory where the FT state should be saved.
            Must be available for all training jobs. NOTE: Beware that PTL/NeMo can move files written directly to `trainer.log_dir`.
            Defaults to None, in which case it defaults to `trainer.log_dir/ft_state/`.
        logger_name (Optional[str], optional): Logger name to be used.
            Defaults to "nemo_logger.FaultToleranceCallback".
    """
```             

### 2. Implementing auto-resume

Auto-resume is a feature that simplifies running a training consisting of multiple subsequent training jobs. 

NOTE: Auto-resume is not a part of the FT package. It is entirely implemented in a launcher script and the `FaultToleranceCallback`. 

`FaultToleranceCallback` exposes an "interface" that allows implementing an auto-resume launcher script.  
Specifically, if `autoresume=True` the FT callback creates a special marker file when a training is completed.  
The marker file location is expected to be set in the `FAULT_TOL_FINISHED_FLAG_FILE` environment variable.

The following mechanism can be used to implement an auto-resuming launcher script:
- Launcher script starts ranks with `ft_launcher`
- `FAULT_TOL_FINISHED_FLAG_FILE` should be passed to rank processes
- When a `ft_launcher` exits, a launcher script checks if the `FAULT_TOL_FINISHED_FLAG_FILE` file was created.
    - If `FAULT_TOL_FINISHED_FLAG_FILE` exists, the auto-resume loop can be broken, as the training is completed.
    - If `FAULT_TOL_FINISHED_FLAG_FILE` does not exist, the continuation job can be issued
        (other conditions can be checked e.g. if the maximum number of failures is not reached).

## Straggler Detection integration guide

### Include `plt_resiliency.StragglerDetectionCallback` in a PTL trainer callbacks. 

```
straggler_cb_args = dict(
    report_time_interval=300.0,
    calc_relative_gpu_perf=True,
    calc_individual_gpu_perf=True,
    num_gpu_perf_scores_to_log=3,
    gpu_relative_perf_threshold=0.7,
    gpu_individual_perf_threshold=0.7,
    stop_if_detected=False,
    logger_name="test_logger",
)

straggler_det_cb = StragglerDetectionCallback(**cb_args)

trainer = pl.Trainer(
    ...
    callbacks=[..., straggler_det_cb],
)
```

`StragglerDetectionCallback` initialization params:

```
def __init__(
    self,
    report_time_interval: float,
    calc_relative_gpu_perf: bool,
    calc_individual_gpu_perf: bool,
    num_gpu_perf_scores_to_log: int,
    gpu_relative_perf_threshold: float,
    gpu_individual_perf_threshold: float,
    stop_if_detected: bool,
    logger_name: Optional[str] = "nemo_logger.StragglerDetectionCallback",
):
    """
    Initialize straggler detection callback instance.

    Args:
        report_time_interval (float): Interval [seconds] of the straggler check
        calc_relative_gpu_perf (bool): Calculate relative GPU performance
        calc_individual_gpu_perf (bool): Calculate individual GPU performance
        num_gpu_perf_scores_to_log (int): How many best and worst scores to log (0 - does not log periodically, but only if stragglers are detected)
        gpu_relative_perf_threshold (float): Threshold for relative GPU performance scores
        gpu_individual_perf_threshold (float): Threshold for individual GPU performance scores
        stop_if_detected (bool): Set to True, to terminate the workload if stragglers are detected
        logger_name (Optional[str], optional): Defaults to "nemo_logger.StragglerDetectionCallback".

    Raises:
        ValueError: If invalid config was provided.
    """
```

More info on straggler detection can be found in the straggler package's README.
