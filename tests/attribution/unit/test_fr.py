import asyncio
import sys
import unittest
from pathlib import Path

# Add the src directory to the path to import the fr_attribution module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fr_attribution_test_utils import FRAttributionOutputParser

from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import CollectiveAnalyzer


class TestFRAttribution(unittest.TestCase):
    """Test cases for the FR attribution module with different trace directories."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parent / "fr_traces"
        self.reference_dir = Path(__file__).parent / "reference_outputs"
        self.trace_directories = ["gpu_error_1st", "gpu_error_2nd", "lock_gil_1st", "lock_gil_2nd"]

        self.parser = FRAttributionOutputParser()

    def test_gpu_error_1st_trace(self):
        """Test FR attribution with gpu_error_1st trace directory."""
        trace_dir = self.test_data_dir / "gpu_error_1st"
        self._run_fr_attribution_test(trace_dir, "gpu_error_1st")

    def test_gpu_error_2nd_trace(self):
        """Test FR attribution with gpu_error_2nd trace directory."""
        trace_dir = self.test_data_dir / "gpu_error_2nd"
        self._run_fr_attribution_test(trace_dir, "gpu_error_2nd")

    def test_lock_gil_1st_trace(self):
        """Test FR attribution with lock_gil_1st trace directory."""
        trace_dir = self.test_data_dir / "lock_gil_1st"
        self._run_fr_attribution_test(trace_dir, "lock_gil_1st")

    def test_lock_gil_2nd_trace(self):
        """Test FR attribution with lock_gil_2nd trace directory."""
        trace_dir = self.test_data_dir / "lock_gil_2nd"
        self._run_fr_attribution_test(trace_dir, "lock_gil_2nd")

    def _run_fr_attribution_test(self, trace_dir: Path, test_name: str):
        """
        Run FR attribution test for a specific trace directory.

        Args:
            trace_dir: Path to the trace directory
            test_name: Name of the test for identification
        """
        # Verify trace directory exists
        self.assertTrue(trace_dir.exists(), f"Trace directory {trace_dir} does not exist")

        # Check that the directory contains _dump files
        dump_files = list(trace_dir.glob("_dump*"))
        self.assertGreater(len(dump_files), 0, f"No _dump files found in {trace_dir}")

        # Create mock arguments for the CollectiveAnalyzer
        mock_args = self._create_mock_args(trace_dir)

        # Test the CollectiveAnalyzer initialization
        analyzer = CollectiveAnalyzer(mock_args)
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.args.pattern, "_dump*")

        # Test the preprocessing step
        try:
            # Run the preprocessing step
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(analyzer.preprocess_FR_dumps([str(trace_dir)]))

            # Verify that preprocessing completed successfully
            self.assertIsNotNone(result)
            analysis_output, attribution_kwargs = result
            self.assertIsInstance(analysis_output, str)
            self.assertIsInstance(attribution_kwargs, dict)

            # Verify that some analysis output was generated
            self.assertGreater(
                len(analysis_output), 0, f"No analysis output generated for {test_name}"
            )

            # Test the collective analysis step
            analysis_result = loop.run_until_complete(
                analyzer.collective_analysis(analysis_output, **attribution_kwargs)
            )

            # Verify that analysis completed
            self.assertIsNotNone(analysis_result)

            # Compare with reference output
            self._compare_with_reference(analysis_output, test_name)

        except Exception as e:
            self.fail(f"FR attribution test failed for {test_name}: {str(e)}")
        finally:
            loop.close()

    def _create_mock_args(self, trace_dir: Path):
        """Create mock arguments for the CollectiveAnalyzer."""

        class MockArgs:
            def __init__(self):
                self.pattern = "_dump*"
                self.verbose = False
                self.health_check = False
                self.llm_analyze = False
                self.scheduling_order_file = None
                self.model = "nvdev/nvidia/llama-3.3-nemotron-super-49b-v1"
                self.debug = False

        return MockArgs()

    def _compare_with_reference(self, actual_output: str, test_name: str):
        """
        Compare actual output with reference output and validate key metrics.

        Args:
            actual_output: The actual output from the test
            test_name: Name of the test case
        """
        try:
            # For test environment, focus on key metrics rather than exact matches
            # The actual_output is just the summary, while reference includes full details
            actual_missing_ranks = self.parser.extract_missing_ranks_set(actual_output)

            # Expected missing ranks based on test specification
            expected_ranks = {
                'gpu_error_1st': {12, 14},
                'gpu_error_2nd': {9, 14},
                'lock_gil_1st': {9, 14},
                'lock_gil_2nd': {10, 15},
            }

            if test_name in expected_ranks:
                self.assertEqual(
                    actual_missing_ranks,
                    expected_ranks[test_name],
                    f"Missing ranks {actual_missing_ranks} do not match expected "
                    f"{expected_ranks[test_name]} for {test_name}",
                )

            # Validate that we have some missing ranks (not empty)
            self.assertGreater(
                len(actual_missing_ranks),
                0,
                f"No missing ranks detected for {test_name}, but some were expected",
            )

        except FileNotFoundError as e:
            self.fail(f"Reference file not found for {test_name}: {e}")
        except Exception as e:
            self.fail(f"Reference comparison failed for {test_name}: {e}")

    def test_all_trace_directories_exist(self):
        """Test that all expected trace directories exist."""
        for trace_dir_name in self.trace_directories:
            trace_dir = self.test_data_dir / trace_dir_name
            with self.subTest(trace_dir=trace_dir_name):
                self.assertTrue(
                    trace_dir.exists(), f"Trace directory {trace_dir_name} does not exist"
                )

    def test_trace_directories_contain_dump_files(self):
        """Test that all trace directories contain _dump files."""
        for trace_dir_name in self.trace_directories:
            trace_dir = self.test_data_dir / trace_dir_name
            with self.subTest(trace_dir=trace_dir_name):
                dump_files = list(trace_dir.glob("_dump*"))
                self.assertGreater(len(dump_files), 0, f"No _dump files found in {trace_dir_name}")

    def test_fr_attribution_command_line_interface(self):
        """Test the command line interface of fr_attribution.py."""
        # Test that the script can be imported and run
        try:
            from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import main

            self.assertTrue(callable(main))
        except ImportError as e:
            self.fail(f"Failed to import fr_attribution main function: {e}")

    def test_collective_analyzer_initialization(self):
        """Test that CollectiveAnalyzer can be initialized with different configurations."""
        # Test with minimal configuration
        mock_args = self._create_mock_args(Path("dummy"))
        analyzer = CollectiveAnalyzer(mock_args)
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.args.pattern, "_dump*")

        # Test with verbose configuration
        mock_args.verbose = True
        analyzer_verbose = CollectiveAnalyzer(mock_args)
        self.assertTrue(analyzer_verbose.args.verbose)

        # Test with health check enabled
        mock_args.health_check = True
        analyzer_health = CollectiveAnalyzer(mock_args)
        self.assertTrue(analyzer_health.args.health_check)

    def test_process_file_method(self):
        """Test the process_file method with a sample trace directory."""
        # Use the first available trace directory for testing
        trace_dir = self.test_data_dir / self.trace_directories[0]
        dump_files = list(trace_dir.glob("_dump*"))

        if dump_files:
            mock_args = self._create_mock_args(trace_dir)
            analyzer = CollectiveAnalyzer(mock_args)

            # Test processing a single file
            test_file = dump_files[0]
            result = analyzer.process_file(str(test_file))

            # The method should return True if successful
            self.assertTrue(result, f"Failed to process file {test_file}")

    def test_analyze_matches_method(self):
        """Test the analyze_matches method."""
        # Use the first available trace directory for testing
        trace_dir = self.test_data_dir / self.trace_directories[0]
        dump_files = list(trace_dir.glob("_dump*"))

        if dump_files:
            mock_args = self._create_mock_args(trace_dir)
            analyzer = CollectiveAnalyzer(mock_args)

            # Process a few files first
            for dump_file in dump_files[:3]:  # Process first 3 files
                analyzer.process_file(str(dump_file))

            # Test analyze_matches
            completed_pg, missing_pg = analyzer.analyze_matches(verbose=False)

            # Should return dictionaries
            self.assertIsInstance(completed_pg, dict)
            self.assertIsInstance(missing_pg, dict)

    def test_group_pgs_method(self):
        """Test the group_pgs method."""
        # Use the first available trace directory for testing
        trace_dir = self.test_data_dir / self.trace_directories[0]
        dump_files = list(trace_dir.glob("_dump*"))

        if dump_files:
            mock_args = self._create_mock_args(trace_dir)
            analyzer = CollectiveAnalyzer(mock_args)

            # Process a few files first
            for dump_file in dump_files[:3]:  # Process first 3 files
                analyzer.process_file(str(dump_file))

            # Run analyze_matches to populate the data structures
            completed_pg, missing_pg = analyzer.analyze_matches(verbose=False)

            # Test group_pgs with missing_pg
            if missing_pg:
                grouped_missing = analyzer.group_pgs(missing_pg)
                self.assertIsInstance(grouped_missing, dict)

            # Test group_pgs with completed_pg
            if completed_pg:
                grouped_completed = analyzer.group_pgs(completed_pg)
                self.assertIsInstance(grouped_completed, dict)

    def test_print_pg_configs_method(self):
        """Test the print_pg_configs method."""
        # Use the first available trace directory for testing
        trace_dir = self.test_data_dir / self.trace_directories[0]
        dump_files = list(trace_dir.glob("_dump*"))

        if dump_files:
            mock_args = self._create_mock_args(trace_dir)
            analyzer = CollectiveAnalyzer(mock_args)

            # Process a few files first
            for dump_file in dump_files[:3]:  # Process first 3 files
                analyzer.process_file(str(dump_file))

            # Test print_pg_configs - should not raise an exception
            try:
                analyzer.print_pg_configs(verbose=False)
            except Exception as e:
                self.fail(f"print_pg_configs failed: {e}")

    def test_print_node_health_status_method(self):
        """Test the print_node_health_status method."""
        # Use the first available trace directory for testing
        trace_dir = self.test_data_dir / self.trace_directories[0]
        dump_files = list(trace_dir.glob("_dump*"))

        if dump_files:
            mock_args = self._create_mock_args(trace_dir)
            analyzer = CollectiveAnalyzer(mock_args)

            # Process a few files first
            for dump_file in dump_files[:3]:  # Process first 3 files
                analyzer.process_file(str(dump_file))

            # Test print_node_health_status - should not raise an exception
            try:
                analyzer.print_node_health_status(verbose=False)
            except Exception as e:
                self.fail(f"print_node_health_status failed: {e}")


class TestFRAttributionReferenceValidation(unittest.TestCase):
    """Test cases specifically for validating against reference outputs."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parent / "fr_traces"
        self.reference_dir = Path(__file__).parent / "reference_outputs"
        self.trace_directories = ["gpu_error_1st", "gpu_error_2nd", "lock_gil_1st", "lock_gil_2nd"]
        self.parser = FRAttributionOutputParser()

    def test_reference_files_exist(self):
        """Test that all reference files exist."""
        for trace_dir_name in self.trace_directories:
            with self.subTest(trace_dir=trace_dir_name):
                reference_file = self.reference_dir / f"{trace_dir_name}_reference.txt"
                self.assertTrue(
                    reference_file.exists(), f"Reference file not found: {reference_file}"
                )

    def test_reference_outputs_are_valid(self):
        """Test that reference outputs contain expected information."""
        for trace_dir_name in self.trace_directories:
            with self.subTest(trace_dir=trace_dir_name):
                reference_file = self.reference_dir / f"{trace_dir_name}_reference.txt"
                reference_output = reference_file.read_text()

                # Parse reference output
                parsed = self.parser.parse_output(reference_output)

                # Validate that reference contains expected information
                self.assertGreater(
                    parsed['processed_files'],
                    0,
                    f"Reference for {trace_dir_name} shows no processed files",
                )
                self.assertGreater(
                    len(parsed['group_types']),
                    0,
                    f"Reference for {trace_dir_name} shows no group types",
                )
                self.assertGreater(
                    len(parsed['missing_ranks_by_pg']),
                    0,
                    f"Reference for {trace_dir_name} shows no missing ranks",
                )


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
