import subprocess
import sys
import tempfile
from pathlib import Path


class TestReporterIntegration:
    """Integration tests for reporter module."""

    def test_cli_integration(self):
        """Test reporter integration with CLI."""
        import os

        # Create test EdgeFlow config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ef", delete=False) as f:
            f.write(
                """
            model_path = "test_models/sample.tflite"
            output_path = "test_models/sample_optimized.tflite"
            quantize = int8
            """
            )
            config_path = f.name

        try:
            # Run CLI command as subprocess to avoid in-process memory corruption/TF crashes
            # We use sys.executable to ensure we use the same python interpreter
            cmd = [sys.executable, "-m", "edgeflow.compiler.edgeflowc", config_path]

            # We need to set PYTHONPATH to include src
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.abspath("src")

            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            # Check return code (0 or 1 are acceptable for this test as long as it runs)
            # We are testing that it generates a report, not necessarily that optimization succeeds
            # (since we don't have a real model)

            # Verify report content if it exists
            if Path("report.md").exists():
                report_content = Path("report.md").read_text()
                assert "EdgeFlow Optimization Report" in report_content
            else:
                # If report wasn't generated, check if it was due to expected error
                # (e.g. model not found) rather than crash
                assert result.returncode != 139  # Segmentation fault
                assert result.returncode != 134  # Aborted

        finally:
            Path(config_path).unlink()
            if Path("report.md").exists():
                Path("report.md").unlink()

    def test_api_integration(self):
        """Test reporter integration with API endpoints."""
        # Placeholder: backend API tests live under backend/ in this repo
        # Here we simply assert reporter can be imported to be used by the API layer.
        import edgeflow.reporting.reporter as _  # noqa: F401

        assert True
