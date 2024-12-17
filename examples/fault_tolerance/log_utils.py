# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
import sys

import dist_utils


def setup_logging(log_all_ranks=True, filename=os.devnull, filemode='w'):
    """
    Configures logging.
    By default logs from all workers are printed to the stderr, entries are
    prefixed with "N: " where N is the rank of the worker. Logs printed to the stderr don't include timestaps.
    Full logs with timestamps are saved to the log_file file.
    """

    class RankFilter(logging.Filter):
        def __init__(self, rank, log_all_ranks):
            self.rank = rank
            self.log_all_ranks = log_all_ranks

        def filter(self, record):
            record.rank = self.rank
            if self.log_all_ranks:
                return True
            else:
                return self.rank == 0

    rank = dist_utils.get_rank()
    rank_filter = RankFilter(rank, log_all_ranks)

    if log_all_ranks:
        logging_format = f"%(asctime)s - %(levelname)s - {rank} - %(message)s"
    else:
        logging_format = "%(asctime)s - %(levelname)s - %(message)s"
        if rank != 0:
            filename = os.devnull

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()

    logging.basicConfig(
        level=logging.DEBUG,
        format=logging_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=filename,
        filemode=filemode,
    )
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.DEBUG)
    if log_all_ranks:
        formatter = logging.Formatter(f'{rank}: %(message)s')
    else:
        formatter = logging.Formatter('%(message)s')
    stderr.setFormatter(formatter)
    logging.getLogger('').addHandler(stderr)
    logging.getLogger('').addFilter(rank_filter)
