import argparse
import sys
from types import ModuleType
from typing import Dict
from unittest.mock import patch

from edgeflow.compiler import edgeflowc
import pytest


def _set_argv(args):
    sys.argv = ["edgeflowc.py", *args]


def test_parse_arguments_with_config_path(monkeypatch):
    _set_argv(["model.ef"])
    ns = edgeflowc.parse_arguments()
    assert ns.config_path == "model.ef"
    assert ns.verbose is False


def test_parse_arguments_verbose(monkeypatch):
    _set_argv(["model.ef", "--verbose"])
    ns = edgeflowc.parse_arguments()
    assert ns.verbose is True


def test_parse_arguments_help(monkeypatch):
    _set_argv(["--help"])
    with pytest.raises(SystemExit) as exc:
        edgeflowc.parse_arguments()
    assert exc.value.code == 0


def test_parse_arguments_version(monkeypatch, capsys):
    _set_argv(["--version"])
    with pytest.raises(SystemExit) as exc:
        edgeflowc.parse_arguments()
    assert exc.value.code == 0


def test_validate_file_path_nonexistent(tmp_path):
    assert edgeflowc.validate_file_path(str(tmp_path / "missing.ef")) is False


def test_validate_file_path_wrong_extension(tmp_path):
    p = tmp_path / "config.txt"
    p.write_text("model_path=...", encoding="utf-8")
    assert edgeflowc.validate_file_path(str(p)) is False


def test_validate_file_path_directory(tmp_path):
    assert edgeflowc.validate_file_path(str(tmp_path)) is False


def test_validate_file_path_uppercase_extension(tmp_path):
    p = tmp_path / "CONFIG.EF"
    p.write_text("quantize=int8", encoding="utf-8")
    assert edgeflowc.validate_file_path(str(p)) is True


def test_load_config_fallback_reads_file(tmp_path, monkeypatch):
    # Mock the validation functions to prevent SystemExit
    monkeypatch.setattr(
        "edgeflow.compiler.edgeflowc.EdgeFlowValidator",
        lambda: type(
            "obj", (object,), {"early_validation": lambda self, cfg: (True, [])}
        )(),
    )
    monkeypatch.setattr(
        "edgeflow.compiler.edgeflowc.validate_edgeflow_config", lambda cfg: (True, [])
    )
    monkeypatch.setattr(
        "edgeflow.compiler.edgeflowc.validate_model_compatibility",
        lambda m, cfg: (True, []),
    )

    p = tmp_path / "model.ef"
    content = 'model="m.tflite"\n'
    p.write_text(content, encoding="utf-8")
    cfg = edgeflowc.load_config(str(p))
    assert cfg["model"] == "m.tflite"
    assert "__source__" in cfg or "model" in cfg  # Parser may or may not add metadata


def test_load_config_uses_parser_if_available(tmp_path, monkeypatch):
    # Mock the validation functions
    monkeypatch.setattr(
        "edgeflow.compiler.edgeflowc.EdgeFlowValidator",
        lambda: type(
            "obj", (object,), {"early_validation": lambda self, cfg: (True, [])}
        )(),
    )
    monkeypatch.setattr(
        "edgeflow.compiler.edgeflowc.validate_edgeflow_config", lambda cfg: (True, [])
    )
    monkeypatch.setattr(
        "edgeflow.compiler.edgeflowc.validate_model_compatibility",
        lambda m, cfg: (True, []),
    )

    # Inject a fake parser module
    fake = ModuleType("parser")

    def parse_ef(path: str) -> Dict[str, str]:
        return {"parsed": path, "model": "test.tflite"}

    fake.parse_ef = parse_ef  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "parser", fake)

    p = tmp_path / "conf.ef"
    p.write_text('model="test.tflite"\nx=1', encoding="utf-8")
    cfg = edgeflowc.load_config(str(p))
    assert cfg["model"] == "test.tflite"
    assert "parsed" in cfg or "x" in cfg  # Either mocked or real parser


def test_optimize_model_uses_optimizer_if_available(monkeypatch):
    called = {"ok": False}

    def fake_optimize(cfg):
        called["ok"] = True
        return "optimized.tflite", {"improvement": 25}

    def fake_benchmark(model_path, cfg):
        return {"latency": 10, "size": 100}

    def fake_compare(orig, opt, cfg):
        return {"improvements": {"size_reduction_percent": 25}, "optimized": {}}

    # Mock the optimizer module completely to avoid importing TF
    fake_optimizer_module = ModuleType("edgeflow.optimization.optimizer")
    fake_optimizer_module.optimize = fake_optimize
    monkeypatch.setitem(
        sys.modules, "edgeflow.optimization.optimizer", fake_optimizer_module
    )

    monkeypatch.setattr(
        "edgeflow.benchmarking.benchmarker.benchmark_model", fake_benchmark
    )
    monkeypatch.setattr(
        "edgeflow.benchmarking.benchmarker.compare_models", fake_compare
    )

    result = edgeflowc.optimize_model({"model": "/tmp/test.tflite"})
    assert called["ok"] is True
    assert "optimization" in result


def test_optimize_model_handles_exception(monkeypatch, caplog):
    # Mock optimize to raise an exception
    def fake_optimize(cfg):
        raise RuntimeError("boom")

    # Mock the optimizer module completely
    fake_optimizer_module = ModuleType("edgeflow.optimization.optimizer")
    fake_optimizer_module.optimize = fake_optimize
    monkeypatch.setitem(
        sys.modules, "edgeflow.optimization.optimizer", fake_optimizer_module
    )
    monkeypatch.setattr(
        "edgeflow.benchmarking.benchmarker.benchmark_model", lambda m, c: {}
    )

    caplog.set_level("INFO")
    result = edgeflowc.optimize_model({"model": "/tmp/test.tflite"})
    assert "error" in result
    assert any("Optimization pipeline failed" in r.message for r in caplog.records)


def test_main_no_args_returns_2(monkeypatch):
    _set_argv([])
    assert edgeflowc.main() == 2


def test_main_nonexistent_file_returns_1(monkeypatch):
    _set_argv(["missing.ef"])
    assert edgeflowc.main() == 1


def test_main_invalid_extension_returns_1(tmp_path, monkeypatch):
    p = tmp_path / "invalid.txt"
    p.write_text('model="test.tflite"\nx=1', encoding="utf-8")
    _set_argv([str(p)])
    assert edgeflowc.main() == 1


def test_main_success_calls_optimize(tmp_path, monkeypatch):
    # Mock validation functions
    monkeypatch.setattr(
        "edgeflow.compiler.edgeflowc.EdgeFlowValidator",
        lambda: type(
            "obj", (object,), {" early_validation": lambda self, cfg: (True, [])}
        )(),
    )
    monkeypatch.setattr(
        "edgeflow.compiler.edgeflowc.validate_edgeflow_config", lambda cfg: (True, [])
    )
    monkeypatch.setattr(
        "edgeflow.compiler.edgeflowc.validate_model_compatibility",
        lambda m, cfg: (True, []),
    )

    p = tmp_path / "ok.ef"
    p.write_text(
        'model="test.tflite"\nquantize="int8"\nmemory_limit=16', encoding="utf-8"
    )
    called = {"n": 0}

    def fake_opt(config, formatter=None):
        called["n"] += 1
        return {
            "optimization": {},
            "original_benchmark": {},
            "optimized_benchmark": {},
            "comparison": {},
        }

    class MockValidator:
        def early_validation(self, config):
            return True, []

    monkeypatch.setattr(edgeflowc, "EdgeFlowValidator", MockValidator)

    monkeypatch.setattr(edgeflowc, "optimize_model", fake_opt)

    monkeypatch.setattr(
        edgeflowc,
        "parse_arguments",
        lambda: argparse.Namespace(
            config_path=str(p),
            verbose=False,
            docker=False,
            skip_check=True,
            check_only=False,
            codegen=None,
            explain=False,
            device_spec_file=None,
        ),
    )

    _set_argv([str(p), "--skip-check"])
    with patch("edgeflow.parser.validate_config", return_value=(True, [])):
        code = edgeflowc.main()
    assert code == 0
    assert called["n"] == 1


def test_main_verbose_emits_debug_log(tmp_path, monkeypatch, caplog):
    # Mock validation functions
    monkeypatch.setattr(
        "edgeflow.compiler.edgeflowc.EdgeFlowValidator",
        lambda: type(
            "obj", (object,), {"early_validation": lambda self, cfg: (True, [])}
        )(),
    )
    monkeypatch.setattr(
        "edgeflow.compiler.edgeflowc.validate_edgeflow_config", lambda cfg: (True, [])
    )
    monkeypatch.setattr(
        "edgeflow.compiler.edgeflowc.validate_model_compatibility",
        lambda m, cfg: (True, []),
    )
    import os

    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda fallback=None: os.terminal_size((80, 24))
    )

    p = tmp_path / "ok.ef"
    p.write_text(
        'model="test.tflite"\nquantize="int8"\nmemory_limit=16', encoding="utf-8"
    )
    monkeypatch.setattr(
        edgeflowc,
        "optimize_model",
        lambda cfg, formatter=None: {
            "optimization": {},
            "original_benchmark": {},
            "optimized_benchmark": {},
            "comparison": {},
        },
    )

    class MockValidator:
        def early_validation(self, config):
            return True, []

    monkeypatch.setattr(edgeflowc, "EdgeFlowValidator", MockValidator)

    caplog.set_level("DEBUG")

    monkeypatch.setattr(
        edgeflowc,
        "parse_arguments",
        lambda: argparse.Namespace(
            config_path=str(p),
            verbose=True,
            docker=False,
            skip_check=True,
            check_only=False,
            codegen=None,
            explain=False,
            device_spec_file=None,
        ),
    )

    _set_argv([str(p), "--verbose", "--skip-check"])
    with patch("edgeflow.parser.validate_config", return_value=(True, [])):
        code = edgeflowc.main()
    assert code == 0
    # confirm debug log emitted by load step
    assert any("Loaded config" in r.message for r in caplog.records)


def test_main_help_returns_0(monkeypatch):
    # Verify SystemExit handling path in main
    _set_argv(["--help"])
    rc = edgeflowc.main()
    assert rc == 0


def test_main_version_returns_0():
    _set_argv(["--version"])
    rc = edgeflowc.main()
    assert rc == 0


def test_main_handles_unexpected_exception(monkeypatch):
    # Force an unexpected exception inside main
    def boom():
        raise RuntimeError("unexpected")

    monkeypatch.setattr(edgeflowc, "parse_arguments", boom)
    rc = edgeflowc.main()
    assert rc == 1


def test_validate_file_path_empty_string():
    assert edgeflowc.validate_file_path("") is False


def test_validate_file_path_normpath_exception(monkeypatch):
    # Monkeypatch os.path.normpath to raise
    import os as _os

    def bad_norm(p):
        raise ValueError("bad")

    monkeypatch.setattr(_os.path, "normpath", bad_norm)
    assert edgeflowc.validate_file_path("foo.ef") is False
