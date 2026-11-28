# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A re-raised worker exception names its class, not its cause.

PyTorch catches an exception inside a DataLoader worker and re-raises it in the
parent as "Caught FileNotFoundError in DataLoader worker process 0.", with the
worker's own traceback printed below under "Original Traceback". The summary
repeats the class and nothing else - it names no file - so closing the
traceback there publishes an identity that cannot distinguish one missing
shard from another.
"""

from nvidia_resiliency_ext.attribution.restart_agent.l0 import build_l0_bundle

RANK = "880: [rank880]: "


def _wrapped_log(tmp_path, *, truncated=False):
    body = [
        "0: iteration 10/1000 | loss 2.0",
        RANK + "Traceback (most recent call last):",
        RANK + '  File "/app/train.py", line 1, in <module>',
        RANK + "FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.",
    ]
    if not truncated:
        body += [
            RANK + "Original Traceback (most recent call last):",
            RANK + '  File "/pkg/torch/utils/data/_utils/worker.py", line 349, in _worker_loop',
            RANK + "    data = fetcher.fetch(index)",
            # Another rank interleaves, as it does in a real aggregated log.
            "874: [rank874]:     output_tensor = forward_step(data_iterator, model)",
            RANK + "FileNotFoundError: [Errno 2] No such file or directory: '/data/shard_07.bin'",
        ]
    log = tmp_path / "run_2200100_date_x_cycle0.log"
    log.write_text("\n".join(body) + "\n")
    return log


def test_the_cause_is_preferred_over_the_wrapper(tmp_path):
    primary = build_l0_bundle(str(_wrapped_log(tmp_path))).deterministic_primary_candidate

    assert primary is not None
    quote = primary.quote or ""
    assert "shard_07.bin" in quote, "the cause names the file; the wrapper does not"
    assert "Caught FileNotFoundError" not in quote


def test_the_wrapper_is_still_used_when_the_cause_never_arrives(tmp_path):
    """A log cut off before the wrapped traceback still beats the header."""
    log = _wrapped_log(tmp_path, truncated=True)

    primary = build_l0_bundle(str(log)).deterministic_primary_candidate

    assert primary is not None
    assert "Caught FileNotFoundError" in (primary.quote or "")


def test_an_ordinary_traceback_still_closes_on_its_own_summary(tmp_path):
    log = tmp_path / "run_2200100_date_x_cycle0.log"
    log.write_text(
        "0: iteration 10/1000 | loss 2.0\n"
        + RANK
        + "Traceback (most recent call last):\n"
        + RANK
        + '  File "/app/train.py", line 1, in <module>\n'
        + RANK
        + "RuntimeError: CUDA out of memory\n"
    )

    primary = build_l0_bundle(str(log)).deterministic_primary_candidate

    assert primary is not None
    assert "CUDA out of memory" in (primary.quote or "")


CKPT_RANK = "428: [rank428]: "


def test_a_checkpoint_aggregate_is_closed_on_the_rank_that_failed(tmp_path):
    """Checkpoint save/load aggregates per-rank failures and re-raises one
    exception naming only the rank set - "CheckpointException ranks:dict_keys
    ([600, 601])" names no error at all. Its wrapped traceback is tagged with
    the rank the frames came from, which is a header like any other and so
    stopped the search at exactly the line that opens the cause."""
    log = tmp_path / "run_2200100_date_x_cycle0.log"
    log.write_text(
        "0: iteration 10/1000 | loss 2.0\n"
        + CKPT_RANK
        + "Traceback (most recent call last):\n"
        + CKPT_RANK
        + '  File "/app/train.py", line 1, in <module>\n'
        + CKPT_RANK
        + "torch.distributed.checkpoint.api.CheckpointException: "
        "CheckpointException ranks:dict_keys([600, 601])\n"
        + CKPT_RANK
        + "Traceback (most recent call last): (RANK 600)\n"
        + CKPT_RANK
        + '  File "/pkg/torch/distributed/checkpoint/utils.py", line 276, in all_gather\n'
        + "874: [rank874]:     output_tensor = forward_step(data_iterator, model)\n"
        + CKPT_RANK
        + "OSError: [Errno 5] Input/output error: '/ckpt/iter_000100/shard.pt'\n"
    )

    primary = build_l0_bundle(str(log)).deterministic_primary_candidate

    assert primary is not None
    quote = primary.quote or ""
    assert "Input/output error" in quote, "the rank that failed names the error"
    assert "ranks:dict_keys" not in quote


def test_an_empty_checkpoint_aggregate_keeps_the_wrapper(tmp_path):
    """``ranks:dict_keys([])`` means no rank reported a sub-failure, so there
    is no cause to prefer and the wrapper is the best anchor available."""
    log = tmp_path / "run_2200100_date_x_cycle0.log"
    log.write_text(
        "0: iteration 10/1000 | loss 2.0\n"
        + CKPT_RANK
        + "Traceback (most recent call last):\n"
        + CKPT_RANK
        + '  File "/app/train.py", line 1, in <module>\n'
        + CKPT_RANK
        + "torch.distributed.checkpoint.api.CheckpointException: "
        "CheckpointException ranks:dict_keys([])\n"
    )

    primary = build_l0_bundle(str(log)).deterministic_primary_candidate

    assert primary is not None
    assert "CheckpointException" in (primary.quote or "")
