# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nvidia_resiliency_ext.fault_tolerance.utils module."""

import os
from unittest.mock import patch

import pytest

from nvidia_resiliency_ext.fault_tolerance.utils import get_infrastructure_rank


class TestGetInfrastructureRank:
    """Tests for get_infrastructure_rank function."""

    @pytest.fixture(autouse=True)
    def clear_env(self):
        """Clear all environment variables that affect get_infrastructure_rank."""
        env_vars = [
            'SLURM_TOPOLOGY_ADDR',
            'SLURM_TOPOLOGY_ADDR_PATTERN',
            'SLURM_TOPOLOGY_NODES_PER_BLOCK',
            'NVRX_INFRA_RANK_FROM_NODENAME',
            'SLURMD_NODENAME',
            'CROSS_SLURM_PROCID',
            'SLURM_PROCID',
            'SLURM_ARRAY_TASK_ID',
            'SLURM_NNODES',
            'SLURM_JOB_NUM_NODES',
            'SLURM_JOB_ID',
            'GROUP_RANK',
        ]
        original_values = {}
        for var in env_vars:
            original_values[var] = os.environ.pop(var, None)

        yield

        # Restore original values
        for var, value in original_values.items():
            if value is not None:
                os.environ[var] = value
            else:
                os.environ.pop(var, None)

    def test_slurm_topology_addr_with_block_node_pattern(self):
        """Test infrastructure rank from SLURM_TOPOLOGY_ADDR with block.node pattern."""
        os.environ['SLURM_TOPOLOGY_ADDR'] = 'block5.node10'
        os.environ['SLURM_TOPOLOGY_ADDR_PATTERN'] = 'block.node'

        rank = get_infrastructure_rank()
        # block=5, node=10, multiplier=10^10 -> 5*10000000000 + 10 = 50000000010
        assert rank == 50000000010

    def test_slurm_topology_addr_with_block_node_pattern_case_insensitive(self):
        """Test that SLURM_TOPOLOGY_ADDR_PATTERN is case-insensitive."""
        os.environ['SLURM_TOPOLOGY_ADDR'] = 'block3.node5'
        os.environ['SLURM_TOPOLOGY_ADDR_PATTERN'] = 'BLOCK.NODE'

        rank = get_infrastructure_rank()
        # block=3, node=5, multiplier=10^10 -> 3*10000000000 + 5 = 30000000005
        assert rank == 30000000005

    def test_slurm_topology_addr_block_no_digits_raises_error(self):
        """Test that SLURM_TOPOLOGY_ADDR with block part having no digits raises ValueError."""
        os.environ['SLURM_TOPOLOGY_ADDR'] = 'block.node5'
        os.environ['SLURM_TOPOLOGY_ADDR_PATTERN'] = 'block.node'

        with pytest.raises(ValueError, match="block part.*contains no digits"):
            get_infrastructure_rank()

    def test_slurm_topology_addr_node_no_digits_raises_error(self):
        """Test that SLURM_TOPOLOGY_ADDR with node part having no digits raises ValueError."""
        os.environ['SLURM_TOPOLOGY_ADDR'] = 'block5.node'
        os.environ['SLURM_TOPOLOGY_ADDR_PATTERN'] = 'block.node'

        with pytest.raises(ValueError, match="node part.*contains no digits"):
            get_infrastructure_rank()

    def test_slurm_topology_addr_invalid_format_raises_error(self):
        """Test that SLURM_TOPOLOGY_ADDR with wrong number of parts raises ValueError."""
        os.environ['SLURM_TOPOLOGY_ADDR'] = 'block5'
        os.environ['SLURM_TOPOLOGY_ADDR_PATTERN'] = 'block.node'

        with pytest.raises(ValueError, match="does not match expected format"):
            get_infrastructure_rank()

    def test_slurm_topology_addr_skip_nodename_logic(self):
        """Test that SLURM_TOPOLOGY_ADDR is skipped when skip_nodename_logic=True."""
        os.environ['SLURM_TOPOLOGY_ADDR'] = 'block5.node3'
        os.environ['SLURM_TOPOLOGY_ADDR_PATTERN'] = 'block.node'
        os.environ['SLURM_PROCID'] = '10'

        rank = get_infrastructure_rank(skip_nodename_logic=True)
        # Should use SLURM_PROCID=10 instead of topology (which would be 5*10^10+3)
        assert rank == 10

    def test_nvrx_infra_rank_from_nodename(self):
        """Test infrastructure rank from NVRX_INFRA_RANK_FROM_NODENAME."""
        os.environ['NVRX_INFRA_RANK_FROM_NODENAME'] = '1'
        os.environ['SLURMD_NODENAME'] = 'node123'

        rank = get_infrastructure_rank()
        assert rank == 123

    def test_nvrx_infra_rank_from_nodename_no_slurmd_nodename_raises_error(self):
        """Test that NVRX_INFRA_RANK_FROM_NODENAME without SLURMD_NODENAME raises ValueError."""
        os.environ['NVRX_INFRA_RANK_FROM_NODENAME'] = '1'

        with pytest.raises(ValueError, match="SLURMD_NODENAME environment variable is not set"):
            get_infrastructure_rank()

    def test_nvrx_infra_rank_from_nodename_no_digits_raises_error(self):
        """Test that SLURMD_NODENAME without digits raises ValueError."""
        os.environ['NVRX_INFRA_RANK_FROM_NODENAME'] = '1'
        os.environ['SLURMD_NODENAME'] = 'node'

        with pytest.raises(ValueError, match="contains no digits"):
            get_infrastructure_rank()

    def test_nvrx_infra_rank_from_nodename_skip_nodename_logic(self):
        """Test that NVRX_INFRA_RANK_FROM_NODENAME is skipped when skip_nodename_logic=True."""
        os.environ['NVRX_INFRA_RANK_FROM_NODENAME'] = '1'
        os.environ['SLURMD_NODENAME'] = 'node456'
        os.environ['SLURM_PROCID'] = '20'

        rank = get_infrastructure_rank(skip_nodename_logic=True)
        assert rank == 20  # Should use SLURM_PROCID instead

    def test_cross_slurm_procid(self):
        """Test infrastructure rank from CROSS_SLURM_PROCID."""
        os.environ['CROSS_SLURM_PROCID'] = '42'

        rank = get_infrastructure_rank()
        assert rank == 42

    def test_cross_slurm_procid_takes_precedence_over_slurm_procid(self):
        """Test that CROSS_SLURM_PROCID takes precedence over SLURM_PROCID."""
        os.environ['CROSS_SLURM_PROCID'] = '42'
        os.environ['SLURM_PROCID'] = '10'

        rank = get_infrastructure_rank()
        assert rank == 42

    def test_slurm_array_task_with_nnodes(self):
        """Test infrastructure rank calculation for SLURM job array."""
        os.environ['SLURM_ARRAY_TASK_ID'] = '3'
        os.environ['SLURM_PROCID'] = '2'
        os.environ['SLURM_NNODES'] = '4'

        rank = get_infrastructure_rank()
        # array_task_id * nnodes + procid = 3 * 4 + 2 = 14
        assert rank == 14

    def test_slurm_array_task_with_job_num_nodes(self):
        """Test infrastructure rank calculation with SLURM_JOB_NUM_NODES fallback."""
        os.environ['SLURM_ARRAY_TASK_ID'] = '2'
        os.environ['SLURM_PROCID'] = '1'
        os.environ['SLURM_JOB_NUM_NODES'] = '5'

        rank = get_infrastructure_rank()
        # array_task_id * nnodes + procid = 2 * 5 + 1 = 11
        assert rank == 11

    def test_slurm_array_task_no_nnodes_raises_error(self):
        """Test that SLURM array task without NNODES raises RuntimeError."""
        os.environ['SLURM_ARRAY_TASK_ID'] = '1'
        os.environ['SLURM_PROCID'] = '0'

        with pytest.raises(RuntimeError, match="SLURM_NNODES/SLURM_JOB_NUM_NODES is not defined"):
            get_infrastructure_rank()

    def test_slurm_procid(self):
        """Test infrastructure rank from SLURM_PROCID."""
        os.environ['SLURM_PROCID'] = '7'

        rank = get_infrastructure_rank()
        assert rank == 7

    def test_group_rank(self):
        """Test infrastructure rank from GROUP_RANK fallback."""
        os.environ['GROUP_RANK'] = '15'

        rank = get_infrastructure_rank()
        assert rank == 15

    def test_slurm_procid_takes_precedence_over_group_rank(self):
        """Test that SLURM_PROCID takes precedence over GROUP_RANK."""
        os.environ['SLURM_PROCID'] = '10'
        os.environ['GROUP_RANK'] = '20'

        rank = get_infrastructure_rank()
        assert rank == 10

    def test_slurm_job_id_without_procid_raises_error(self):
        """Test that SLURM_JOB_ID without SLURM_PROCID raises RuntimeError."""
        os.environ['SLURM_JOB_ID'] = '12345'

        with pytest.raises(
            RuntimeError, match="neither CROSS_SLURM_PROCID nor SLURM_PROCID is defined"
        ):
            get_infrastructure_rank()

    def test_no_env_vars_returns_minus_one(self):
        """Test that no environment variables returns -1."""
        rank = get_infrastructure_rank()
        assert rank == -1

    def test_precedence_nodename_over_topology_addr(self):
        """Test that NVRX_INFRA_RANK_FROM_NODENAME takes precedence over SLURM_TOPOLOGY_ADDR."""
        os.environ['SLURM_TOPOLOGY_ADDR'] = 'block1.node2'
        os.environ['SLURM_TOPOLOGY_ADDR_PATTERN'] = 'block.node'
        os.environ['NVRX_INFRA_RANK_FROM_NODENAME'] = '1'
        os.environ['SLURMD_NODENAME'] = 'node999'

        rank = get_infrastructure_rank()
        # Should use nodename: 999 (not topology 1*10^10 + 2 = 10000000002)
        assert rank == 999

    def test_precedence_nodename_over_cross_slurm_procid(self):
        """Test that NVRX_INFRA_RANK_FROM_NODENAME takes precedence over CROSS_SLURM_PROCID."""
        os.environ['NVRX_INFRA_RANK_FROM_NODENAME'] = '1'
        os.environ['SLURMD_NODENAME'] = 'node100'
        os.environ['CROSS_SLURM_PROCID'] = '50'

        rank = get_infrastructure_rank()
        assert rank == 100  # Should use nodename, not cross_slurm_procid

    def test_precedence_cross_slurm_procid_over_array_task(self):
        """Test that CROSS_SLURM_PROCID takes precedence over SLURM array calculation."""
        os.environ['CROSS_SLURM_PROCID'] = '25'
        os.environ['SLURM_ARRAY_TASK_ID'] = '2'
        os.environ['SLURM_PROCID'] = '3'
        os.environ['SLURM_NNODES'] = '4'

        rank = get_infrastructure_rank()
        assert rank == 25  # Should use cross_slurm_procid

    def test_precedence_array_task_over_slurm_procid(self):
        """Test that SLURM array task calculation takes precedence over plain SLURM_PROCID."""
        os.environ['SLURM_ARRAY_TASK_ID'] = '1'
        os.environ['SLURM_PROCID'] = '2'
        os.environ['SLURM_NNODES'] = '3'

        rank = get_infrastructure_rank()
        # Should use array calculation: 1 * 3 + 2 = 5, not just SLURM_PROCID=2
        assert rank == 5

    def test_array_task_zero(self):
        """Test SLURM array task with task ID 0."""
        os.environ['SLURM_ARRAY_TASK_ID'] = '0'
        os.environ['SLURM_PROCID'] = '0'
        os.environ['SLURM_NNODES'] = '2'

        rank = get_infrastructure_rank()
        assert rank == 0

    def test_topology_addr_complex_format(self):
        """Test SLURM_TOPOLOGY_ADDR with complex format extracts digits from each part."""
        os.environ['SLURM_TOPOLOGY_ADDR'] = 'b1l2o3c4k5.n6o7d8e9'
        os.environ['SLURM_TOPOLOGY_ADDR_PATTERN'] = 'block.node'

        rank = get_infrastructure_rank()
        # block part: b1l2o3c4k5 -> digits 12345 -> block=12345
        # node part: n6o7d8e9 -> digits 6789 -> node=6789
        # multiplier=10^10, rank = 12345 * 10000000000 + 6789 = 123450000006789
        assert rank == 123450000006789

    def test_topology_addr_ordering_examples(self):
        """Test specific ordering examples to ensure proper rank calculation with multiplier=10^10."""
        test_cases = [
            ('block5.node3', 50000000003),  # 5*10^10 + 3
            ('block5.node9', 50000000009),  # 5*10^10 + 9
            ('block5.node10', 50000000010),  # 5*10^10 + 10
            ('block6.node2', 60000000002),  # 6*10^10 + 2
        ]

        os.environ['SLURM_TOPOLOGY_ADDR_PATTERN'] = 'block.node'

        for topology_addr, expected_rank in test_cases:
            os.environ['SLURM_TOPOLOGY_ADDR'] = topology_addr
            rank = get_infrastructure_rank()
            assert (
                rank == expected_rank
            ), f"Expected {expected_rank} for {topology_addr}, got {rank}"

        # Verify ordering: all ranks should be in ascending order
        ranks = [tc[1] for tc in test_cases]
        assert ranks == sorted(ranks), f"Ranks should be in ascending order: {ranks}"

    def test_topology_addr_custom_multiplier(self):
        """Test that SLURM_TOPOLOGY_NODES_PER_BLOCK env var overrides default multiplier."""
        os.environ['SLURM_TOPOLOGY_ADDR'] = 'block5.node10'
        os.environ['SLURM_TOPOLOGY_ADDR_PATTERN'] = 'block.node'
        os.environ['SLURM_TOPOLOGY_NODES_PER_BLOCK'] = '1000'

        rank = get_infrastructure_rank()
        # With multiplier=1000: 5*1000 + 10 = 5010
        assert rank == 5010

    def test_topology_addr_node_exceeds_10_digits_raises_error(self):
        """Test that node number >= 10^10 raises ValueError."""
        os.environ['SLURM_TOPOLOGY_ADDR'] = 'block1.node10000000000'  # 10^10
        os.environ['SLURM_TOPOLOGY_ADDR_PATTERN'] = 'block.node'

        with pytest.raises(ValueError, match="Node number.*exceeds maximum supported value"):
            get_infrastructure_rank()

    def test_nodename_complex_format(self):
        """Test SLURMD_NODENAME with letters and digits extracts only digits."""
        os.environ['NVRX_INFRA_RANK_FROM_NODENAME'] = '1'
        os.environ['SLURMD_NODENAME'] = 'compute-node-4-2-1'

        rank = get_infrastructure_rank()
        assert rank == 421  # Digits: 4, 2, 1 -> 421

    @patch('nvidia_resiliency_ext.fault_tolerance.utils.logger')
    def test_logging_for_topology_addr(self, mock_logger):
        """Test that appropriate debug logs are generated for SLURM_TOPOLOGY_ADDR."""
        os.environ['SLURM_TOPOLOGY_ADDR'] = 'block1.node0'
        os.environ['SLURM_TOPOLOGY_ADDR_PATTERN'] = 'block.node'

        rank = get_infrastructure_rank()

        # block=1, node=0, multiplier=10^10 -> 1*10000000000 + 0 = 10000000000
        assert rank == 10000000000
        mock_logger.debug.assert_called_once()
        assert "SLURM_TOPOLOGY_ADDR" in mock_logger.debug.call_args[0][0]

    @patch('nvidia_resiliency_ext.fault_tolerance.utils.logger')
    def test_logging_for_nodename(self, mock_logger):
        """Test that appropriate debug logs are generated for NVRX_INFRA_RANK_FROM_NODENAME."""
        os.environ['NVRX_INFRA_RANK_FROM_NODENAME'] = '1'
        os.environ['SLURMD_NODENAME'] = 'node5'

        rank = get_infrastructure_rank()

        assert rank == 5
        mock_logger.debug.assert_called_once()
        assert "SLURMD_NODENAME" in mock_logger.debug.call_args[0][0]

    @patch('nvidia_resiliency_ext.fault_tolerance.utils.logger')
    def test_logging_for_no_env_vars(self, mock_logger):
        """Test that appropriate debug logs are generated when no env vars are set."""
        rank = get_infrastructure_rank()

        assert rank == -1
        mock_logger.debug.assert_called_once()
        assert "deterministically" in mock_logger.debug.call_args[0][0]
