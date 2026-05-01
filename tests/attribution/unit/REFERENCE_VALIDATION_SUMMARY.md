# FR Attribution Reference Value Validation Summary

## Overview

This document summarizes the implementation of reference value validation for the FR attribution test cases. The test suite now includes comprehensive comparison against reference outputs to ensure consistent and accurate results.

### 1. Reference Output Capture
- **Location**: `/tests/attribution/unit/reference_outputs/`
- **Files**: 
  - `gpu_error_1st_reference.txt`
  - `gpu_error_2nd_reference.txt`
  - `lock_gil_1st_reference.txt`
  - `lock_gil_2nd_reference.txt`

These reference files contain the final summary table returned by the current `CollectiveAnalyzer.preprocess_FR_dumps()` implementation for each test case.

### 2. Test Utility Module
- **File**: `fr_attribution_test_utils.py`
- **Classes**:
  - `FRAttributionOutputParser`: Parses FR attribution output to extract key metrics


### Reference Value Validation
Each test case now validates against reference outputs by checking:

1. **Missing Ranks**: Verifies that the ranks identified by the current wavefront attribution logic match the reference summaries.
   - `gpu_error_1st`: {12}
   - `gpu_error_2nd`: {9, 14}
   - `lock_gil_1st`: {9, 14}
   - `lock_gil_2nd`: {10, 15}

2. **Process Group Analysis**: Ensures process groups are correctly identified and analyzed

3. **Output Consistency**: Validates that the analysis produces meaningful results

## Usage

### Running All Tests
```bash
python -m unittest discover -s tests/attribution/unit -p 'test_fr.py' -v
```
