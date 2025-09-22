# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Monkey patch for PyTorch's c10d_rendezvous_backend to add use_libuv support.

This patch modifies the _create_tcp_store function to accept and use the use_libuv
parameter from RendezvousParameters, allowing users to control whether to use
the libuv backend or the traditional socket backend for TCPStore.

Usage:
    from nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch import apply_c10d_patch
    apply_c10d_patch()
"""

import logging

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)


def _patched_create_tcp_store(params: "RendezvousParameters") -> "TCPStore":  # noqa: F821
    """
    Patched version of _create_tcp_store that supports use_libuv parameter.

    This function is identical to the original _create_tcp_store except it
    extracts and uses the use_libuv parameter from RendezvousParameters.
    """
    import os
    from datetime import timedelta
    from typing import cast

    from torch.distributed import TCPStore
    from torch.distributed.elastic.events import NodeState, construct_and_record_rdzv_event
    from torch.distributed.elastic.rendezvous.api import RendezvousConnectionError
    from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import (
        DEFAULT_PORT,
        _matches_machine_hostname,
        parse_rendezvous_endpoint,
    )

    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=DEFAULT_PORT)

    cfg_is_host = params.get_as_bool("is_host")
    # If the user has explicitly specified whether our process should host the
    # the store, respect it.
    if cfg_is_host is not None:
        is_host = cfg_is_host
    # Otherwise try to determine whether we are the host based on our hostname
    # and IP address.
    else:
        is_host = _matches_machine_hostname(host)

    # The timeout
    read_timeout = cast(int, params.get_as_int("read_timeout", 60))
    if read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")

    # The use_libuv parameter - NEW FUNCTIONALITY
    use_libuv = params.get_as_bool("use_libuv", True)

    # In specific cases we attempt to instantiate the store twice. For details
    # see the explanation in the except clause below.
    for is_server in [is_host, False]:
        try:
            store = TCPStore(
                host,
                port,
                is_master=is_server,
                multi_tenant=True,
                timeout=timedelta(seconds=read_timeout),
                use_libuv=use_libuv,  # NEW PARAMETER
            )

            if is_server:
                msg = f"Process {os.getpid()} hosts the TCP store for the C10d rendezvous backend."
                construct_and_record_rdzv_event(
                    run_id=params.run_id, message=msg, node_state=NodeState.INIT
                )
                logger.info(msg)

            break
        except (ValueError, RuntimeError, TimeoutError) as exc:
            # If we heuristically inferred the value of is_host as True and our
            # first attempt to instantiate the TCP store has failed, try it one
            # more time with is_host set to False. As an edge case there can be
            # more than one process that is part of the same rendezvous on this
            # machine and only one of them will eventually host the store.

            if not is_server or cfg_is_host is not None:
                raise RendezvousConnectionError(
                    "The connection to the C10d store has failed. See inner exception for details."
                ) from exc

    return store  # type: ignore[possibly-undefined]


def apply_c10d_patch():
    """
    Apply the monkey patch to add use_libuv support to c10d_rendezvous_backend.

    This function patches the _create_tcp_store function in the c10d_rendezvous_backend
    module to support the use_libuv parameter.
    """
    try:
        from torch.distributed.elastic.rendezvous import c10d_rendezvous_backend

        # Apply the patch
        c10d_rendezvous_backend._create_tcp_store = _patched_create_tcp_store

        logger.info(
            "Successfully applied c10d_rendezvous_backend monkey patch for use_libuv support"
        )

    except ImportError as e:
        logger.error(f"Failed to import c10d_rendezvous_backend: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to apply c10d monkey patch: {e}")
        raise
