# Straggler Detection Tests
## Running unit tests:
```
python3 -m pytest -rs -x ./tests/unit/
```

### Unit tests coverage:
- `test_cupti_ext.py`: Test API of C++ class `CuptiProfiler`.

- `test_cupti_manager.py`: Test `CuptiManager`, Python class wrapping `CuptiProfiler`.

- `test_data_shared.py`: Ensure size of shared data is reduced after kernel names are mapped into IDs.
- `test_det_section_api.py`: Test `Detector.detection_section` context manager behavior.
- `test_individual_gpu_scores.py`: Verify individual GPU scores values, `rank_to_node` report field. 
- `test_interval_tracker.py`: Test `ReportIntervalTracker` reporting interval estimation functionality.
- `test_name_mapper.py`: Test `NameMapper` API.
- `test_relative_gpu_scores.py`: Same as `test_individual_gpu_scores.py` for relative GPU scores.
- `test_reporting_elapsed.py`: Test `Detector.generate_report_if_interval_elapsed` and `ReportIntervalTracker` functionality.
- `test_reporting.py`: Test `Detector.generate_report` and `Report.identify_stragglers` functionality.
- `test_sections.py`: Simulate straggler sections and verify correct detection based on both individual and relative scores.
- `test_wrap_callables.py`: Test `Detector.wrap_callables` behavior.

### Running multi-GPU tests:
While `test_reporting_elapsed.py` and `test_reporting.py` perform unit tests to check functionalities, the complete testing scenarios are validated when running on multiple GPUs.
```
torchrun --nproc-per-node=8 tests/unit/test_reporting.py
torchrun --nproc-per-node=8 tests/unit/test_reporting_elapsed.py
```


## Running functional tests
Testing straggler detection with various options combinations with various combinations of arguments in multi GPU setting. Below are examples of how to run tests using `torchrun`.

### Example commands:

1. Test with `.generate_if_interval_elapsed` and `gather_on_rank0=False`:
   ```
   torchrun --nproc-per-node=8 tests/func/ddp_test.py --generate_if_elapsed --no_gather_on_rank0
   ```

2. Test with `gather_on_rank0=False` and `scores_to_compute=["relative_perf_scores"]`:
   ```
   torchrun --nproc-per-node=8 tests/func/ddp_test.py --no_gather_on_rank0 --no_indiv_scores
   ```

### Available arguments

- `--no_rel_scores`: Do not compute relative performance scores.
- `--no_indiv_scores`: Do not compute individual performance scores.
- `--no_gather_on_rank0`: Set `gather_on_rank0` to `False`.
- `--use_wrap_forward`: Use wrap callables instead of detection section (default: `False`).
- `--report_iter_interval`: Interval for generating report in iterations (default: `1000`).
- `--generate_if_elapsed`: Generate report if interval elapsed (default: `False`).
- `--report_time_interval`: Time interval for generating report if interval elapsed in seconds (default: `1`).

