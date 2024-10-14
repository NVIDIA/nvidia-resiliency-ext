# Straggler Detection

The **Straggler Detection** package's purpose is to detect slower ranks participating in a PyTorch distributed workload.

## Package Overview

Package users can mark sections of their code as "straggler detection sections". 
Timing information is collected for such sections, which includes CPU (wall-clock) time spent in the section. 
Additionally, CUDA kernels launched in the detection sections can be benchmarked. 

Based on the timings collected, the following performance scores are computed:
- For each detection section, there is a per-rank performance score based on the measured CPU wall-clock time spent in the section.
- [Optionally] All CUDA kernel timing information from all monitored sections is aggregated into a single per-rank GPU performance score.

Performance scores are scalar values from 0.0 (worst) to 1.0 (best), reflecting each rank's performance.
A performance score can be interpreted as the ratio of current performance to reference performance.                                                                                                                                
                                                                                                                                                                                
Depending on the reference used, there are two types of performance scores:                              
- Relative performance score: The best-performing rank in the workload is used as a reference.                                                                                                                               
- Individual performance score: The best historical performance of the rank is used as a reference.                                                                                                                          
                                                                                                                                                                                                                                    
Examples:                                                                                                                                                                                                                           
- If the relative performance score is 0.5, it means that a rank is twice slower than the fastest rank.         
- If the individual performance score is 0.5, it means that a rank is twice slower than its best performance.
                                                                                                                                                                                                                                    
If there are 8 ranks and 2 sections "data_loading" and "fwd" defined, scores might look like this:

```
Relative section performance scores:
{
'data_loading': {0: 0.961, 1: 0.919, 2: 0.941, 3: 0.999, 4: 1.0, 5: 0.847, 6: 0.591, 7: 0.788},
'fwd': {0: 0.994, 1: 0.949, 2: 0.994, 3: 0.998, 4: 1.0, 5: 0.947, 6: 0.991, 7: 0.988}
}
Relative GPU performance scores:
{
0: 0.703, 1: 0.978, 2: 0.689, 3: 0.663, 4: 0.673, 5: 0.925, 6: 0.720, 7: 0.737
}

# NOTE: GPU performance scores are not related to any particular section.

# 
# Individual scores have the same structure as relative scores.
#
```

If the performance score drops below the specified threshold, the corresponding section or GPU is identified as a straggler.         
To detect the stragglers, users should call the reporting function periodically, and check the returned report object for stragglers.

## Installation
- `pip install .`
- `python3 -m pytest -rs -x ./tests/unit/`

Requirements:
- Python >= 3.10
- gcc >= 8.0
- CUDA >= 11.8

## Usage
Sections of the user code that should be monitored for performance ("detection sections") can be defined using:
- `straggler.Detector.detection_section` context manager that allows wrapping a custom code block.
- `straggler.Detector.wrap_callables` method that wraps given callables within `straggler.Detector.detection_section`.

Both methods yield the same results. Users can choose the API that is most convenient for their needs.

If a detection section is monitored for CPU performance, a given section is supposed to run a similar amount of work on all ranks 
(e.g., if a significant amount of additional work was conducted on rank 0, rank 0 would be identified as a straggler).

Captured CUDA kernels can differ between ranks, but there should be some subset common to all ranks.
The simplest approach might be to wrap the model's `forward` call.

Using `straggler.Detector.detection_section`:
```
import nvidia_resiliency_ext.straggler as straggler

straggler.Detector.initialize(
    scores_to_compute=["relative_perf_scores", "individual_perf_scores"],
    gather_on_rank0=True # all ranks results will be available on rank 0
)

for training_iter in range(num_iters):
    
    ...

    with straggler.Detector.detection_section("data_loading", profile_cuda=False):
        data, labels = data_loader.get_next_batch()

    with straggler.Detector.detection_section("fwd", profile_cuda=True):
        out = model(data)
    ...

    if (training_iter % STRAGGLER_REPORT_INTERVAL) == 0:
        report = straggler.Detector.generate_report()
        if rank == 0:
            stragglers = report.identify_stragglers()
            maybe_notify_user(stragglers)

...

straggler.Detector.shutdown()
```

Using `straggler.Detector.wrap_callables`:
```
import nvidia_resiliency_ext.straggler as straggler

straggler.Detector.initialize(
    scores_to_compute=["relative_perf_scores", "individual_perf_scores"],
    gather_on_rank0=True # all ranks results will be available on rank 0
)

straggler.Detector.wrap_callables(
    callable_ids=[
        straggler.CallableId(DataLoaderClass, "get_next_batch", ignored_args=("self",)),
        straggler.CallableId(ModelClass, "forward", ignored_args=("self",)),
    ]
)

for training_iter in range(num_iters):
    
    ...
    
    data, labels = data_loader.get_next_batch()
    out = model(data)

    ...

    if (training_iter % STRAGGLER_REPORT_INTERVAL) == 0:
        report = straggler.Detector.generate_report()
        if rank == 0:
            stragglers = report.identify_stragglers()
            maybe_notify_user(stragglers)

...

straggler.Detector.shutdown()
```

### Alternative reporting
Besides calling `straggler.Detector.generate_report` using construction `if (iter_idx % REPORT_INTERVAL) == 0:`,
(where `REPORT_INTERVAL` is a user-provided value tweaked for a given workload), users can choose to call 
`straggler.Detector.generate_report_if_interval_elapsed` along with each training step.
The report generation will occur approx. within each period specified through `straggler.Detector.initialize(report_time_interval=...)`.


```
import nvidia_resiliency_ext.straggler as straggler

straggler.Detector.initialize(report_time_interval=60, ...) # Report each 1 minute

for training_iter in range(num_iters):

    ...

    # During each training iter
    report = straggler.Detector.generate_report_if_interval_elapsed()
    if report is not None:
        handle_report(report)

    # Note: straggler.Detector.generate_report() works as usual, user can alternatively use:
    # if (iter_idx % REPORT_INTERVAL) == 0:
    #     straggler.Detector.generate_report()

...

straggler.Detector.shutdown()
```

## Reducing the overhead

CUDA kernel profiling imposes some small step time overhead that depends on the workload but generally is expected to be <1%. 
If needed, the amount of overhead can be reduced with `straggler.Detector.initialize(profiling_interval=...)`. 
If `profiling_interval` is > 1, only a fraction of section runs are profiled. 
TODO: provide profiling_interval automatic tuning.

`Detector.generate_report()` overhead depends on the `Detector.initialize` parameters:
- Relative performance score calculation (`scores_to_compute=["relative_perf_scores", ...]`) involves sharing some data between ranks.
- If `gather_on_rank0=True` results from all ranks are gathered on rank 0.

Hence, the following initialization parameters can be used to avoid any inter-rank synchronization:
```
straggler.Detector.initialize(
    scores_to_compute=["individual_perf_scores"],
    gather_on_rank0=False # each rank will report its own results
)
```
In that case, all ranks compute and report their own, individual results only. 

## Known limitations/TODO
- Requires CUDA to be available. TODO: should not require CUDA if GPU performance monitoring is not used. 
- The package user should call `straggler.Detector.generate_report` with suitable frequency (once every few minutes). 
  It should not be called too often as it synchronizes data on all ranks, which can introduce significant overhead. 
