#!/bin/bash

# Verify `ft_test_launchers.sh` output from a given stage
# Usage is `./ft_test_asserts.sh <stage idx>`

set -x
set -o pipefail

: "${FT_CONT_OUT_DIR:?Error: FT_CONT_OUT_DIR is not set or empty}"
: "${LOG_FILE:?Error: LOG_FILE is not set or empty}"

function assert_log_contains {
    expected_str="$1"
    if ! grep -q "${expected_str}" ${FT_CONT_OUT_DIR}/${LOG_FILE}; then
        echo "Expected string not found in logs: ${expected_str}"
        exit 1
    fi
}

function assert_not_in_log {
    not_expected_str="$1"
    if grep -q "${not_expected_str}" ${FT_CONT_OUT_DIR}/${LOG_FILE}; then
        echo "Not expected string found in logs: ${not_expected_str}"
        exit 1
    fi
}

function assert_checkpoint_saved {
    if [ -d "${FT_CONT_OUT_DIR}/default/checkpoints/step*-last" ] ; then
        echo "Expected last checkpoint to be saved, but not found in ${FT_CONT_OUT_DIR}/default/checkpoints/"
        exit 1
    fi
}

function assert_number_of_runs {
    expected_num=$1
    actual_num=$(grep -c "All distributed processes registered." ${FT_CONT_OUT_DIR}/${LOG_FILE})
    if [ "$expected_num" -ne "$actual_num" ]; then
        echo "Expected runs: ${expected_num}, but got ${actual_num}"
        exit 1
    fi
}

function assert_all_launchers_succeeded {
    assert_not_in_log "Some rank(s) exited with non-zero exit code"
}

function assert_launchers_failed {
    assert_log_contains "Some rank(s) exited with non-zero exit code"
}

case "$1" in
    1)
        assert_log_contains "Simulating fault"
        assert_log_contains "FT timeout elapsed"
        assert_checkpoint_saved
        assert_launchers_failed
        ;;
    2)
        assert_log_contains "Time limit reached."
        assert_log_contains "Updated FT timeouts."
        assert_all_launchers_succeeded
        ;;
    3)
        assert_number_of_runs 3
        assert_log_contains "Simulating fault"
        assert_log_contains "FT timeout elapsed"
        assert_launchers_failed
        ;;
    4)
        assert_log_contains "Time limit reached."
        assert_log_contains "Updated FT timeouts."
        assert_all_launchers_succeeded
        ;;
    *)
        echo "Invalid stage for assertions."
        exit 1
        ;;
esac

echo "Assertions for stage $1 passed."
