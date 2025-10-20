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

These reference files contain the complete output from running `fr_attribution.py -p "_dump*" <trace_directory>` for each test case.

### 2. Test Utility Module
- **File**: `fr_attribution_test_utils.py`
- **Classes**:
  - `FRAttributionOutputParser`: Parses FR attribution output to extract key metrics


### Reference Value Validation
Each test case now validates against reference outputs by checking:

1. **Missing Ranks**: Verifies that the correct ranks are identified as missing
   - `gpu_error_1st`: {12, 14}
   - `gpu_error_2nd`: {9, 14}
   - `lock_gil_1st`: {9, 14}
   - `lock_gil_2nd`: {10, 15}

2. **Process Group Analysis**: Ensures process groups are correctly identified and analyzed

3. **Output Consistency**: Validates that the analysis produces meaningful results

## Usage

### Running All Tests
```bash
python -m pytest tests/attribution/unit/test_fr.py -v
```
