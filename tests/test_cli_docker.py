import sys
from types import ModuleType

from edgeflow.compiler import edgeflowc


def _write_cfg(tmp_path, content):
    p = tmp_path / "edgeflow.config"
    p.write_text(content)
    return str(p)


def test_cli_docker_flag(tmp_path, monkeypatch):
    """If --docker is passed, we should invoke DockerManager."""
    # Mock validation functions
    monkeypatch.setattr("edgeflow.compiler.edgeflowc.EdgeFlowValidator", lambda: type('obj', (object,), {'early_validation': lambda self, cfg: (True, [])})(  ))
    monkeypatch.setattr("edgeflow.compiler.edgeflowc.validate_edgeflow_config", lambda cfg: (True, []))
    monkeypatch.setattr("edgeflow.compiler.edgeflowc.validate_model_compatibility", lambda m, cfg: (True, []))
    
    cfg = _write_cfg(tmp_path, 'model="m.tflite"\n')

    # Mock docker_manager module with correct path
    class FakeDM:
        def build_image(self, **kwargs):
            return True

        def run_optimization_pipeline(self, **kwargs):
            return {"success": True}

    def validate_docker_setup():
        return {
            "docker_installed": True,
            "compose_installed": True,
            "docker_running": True,
        }

    monkeypatch.setattr("edgeflow.deployment.docker_manager.DockerManager", FakeDM)
    monkeypatch.setattr("edgeflow.deployment.docker_manager.validate_docker_setup", validate_docker_setup)

    # Mock sys.argv
    monkeypatch.setattr(
        sys, "argv", ["edgeflowc.py", cfg, "--docker"]
    )  # type: ignore[attr-defined]

    rc = edgeflowc.main()
    assert rc == 0


def test_cli_docker_run_failure(tmp_path, monkeypatch):
    """If docker run returns success False, exit with code 1."""
    # Mock validation functions
    monkeypatch.setattr("edgeflow.compiler.edgeflowc.EdgeFlowValidator", lambda: type('obj', (object,), {'early_validation': lambda self, cfg: (True, [])})(  ))
    monkeypatch.setattr("edgeflow.compiler.edgeflowc.validate_edgeflow_config", lambda cfg: (True, []))
    monkeypatch.setattr("edgeflow.compiler.edgeflowc.validate_model_compatibility", lambda m, cfg: (True, []))
    
    cfg = _write_cfg(tmp_path, 'model="m.tflite"\n')

    class FakeDM:
        def build_image(self, **kwargs):
            return True

        def run_optimization_pipeline(self, **kwargs):
            return {"success": False, "error": "boom"}

    def validate_docker_setup():
        return {
            "docker_installed": True,
            "compose_installed": True,
            "docker_running": True,
        }

    monkeypatch.setattr("edgeflow.deployment.docker_manager.DockerManager", FakeDM)
    monkeypatch.setattr("edgeflow.deployment.docker_manager.validate_docker_setup", validate_docker_setup)

    monkeypatch.setattr(
        sys, "argv", ["edgeflowc.py", cfg, "--docker"]
    )  # type: ignore[attr-defined]
    rc = edgeflowc.main()
    assert rc == 1


def test_cli_docker_build_and_run(tmp_path, monkeypatch):
    """Happy-path: docker build (optional) and run succeed -> exit 0."""
    # Mock validation functions
    monkeypatch.setattr("edgeflow.compiler.edgeflowc.EdgeFlowValidator", lambda: type('obj', (object,), {'early_validation': lambda self, cfg: (True, [])})(  ))
    monkeypatch.setattr("edgeflow.compiler.edgeflowc.validate_edgeflow_config", lambda cfg: (True, []))
    monkeypatch.setattr("edgeflow.compiler.edgeflowc.validate_model_compatibility", lambda m, cfg: (True, []))
    
    cfg = _write_cfg(tmp_path, 'model="m.tflite"\n')

    # Fake docker_manager module
    class FakeDM:
        def build_image(self, **kwargs):
            return True

        def run_optimization_pipeline(self, **kwargs):
            return {"success": True, "output_path": "./outputs"}

    def validate_docker_setup():
        return {
            "docker_installed": True,
            "compose_installed": True,
            "docker_running": True,
        }

    monkeypatch.setattr("edgeflow.deployment.docker_manager.DockerManager", FakeDM)
    monkeypatch.setattr("edgeflow.deployment.docker_manager.validate_docker_setup", validate_docker_setup)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "edgeflowc.py",
            cfg,
            "--docker",
            "--docker-build",
            "--docker-tag",
            "mytag",
        ],
    )
    rc = edgeflowc.main()
    assert rc == 0
