"""M0 environment checks only: no market data and no financial models."""

import importlib
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import tempfile
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "plotly": "plotly",
    "rich": "rich",
    "networkx": "networkx",
    "akshare": "akshare",
    "torch": "torch",
    "pytest": "pytest",
}


def test_python_version_and_location():
    assert sys.version.split()[0] == (ROOT / ".python-version").read_text().strip()
    assert sys.version_info[:2] == (3, 13)
    assert platform.machine().upper() == "AMD64"
    assert Path(sys.prefix).resolve() == ROOT / ".venv"
    assert Path(sys.executable).resolve().is_relative_to(ROOT / ".venv")
    assert Path(sys.base_prefix).resolve().is_relative_to(ROOT / ".tools/python")


@pytest.mark.parametrize("distribution,module", PACKAGES.items())
def test_import_and_pinned_version(distribution, module):
    imported = importlib.import_module(module)
    assert Path(imported.__file__).resolve().is_relative_to(ROOT / ".venv")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pins = config["project"]["dependencies"] + config["dependency-groups"]["dev"]
    expected = dict(pin.split("==") for pin in pins)
    assert version(distribution) == expected[distribution]


def test_numpy_scipy_numerics():
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq

    np.random.seed(42)
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    rhs = np.array([9.0, 8.0])
    np.testing.assert_allclose(matrix @ np.linalg.solve(matrix, rhs), rhs, atol=1e-12)
    assert brentq(lambda x: x * x - 2.0, 0.0, 2.0) == pytest.approx(np.sqrt(2), abs=1e-12)
    integral, error = quad(lambda x: x * x, 0.0, 1.0)
    assert integral == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert error < 1e-10


def test_torch_tensor_and_autograd():
    import torch

    torch.manual_seed(42)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    result = (x * x).sum()
    result.backward()
    assert result.item() == 14.0
    torch.testing.assert_close(x.grad, 2 * x.detach(), rtol=0, atol=1e-12)


def test_plotly_self_contained_html(tmp_path):
    from html.parser import HTMLParser

    import plotly.graph_objects as go

    class Scripts(HTMLParser):
        sources = []

        def handle_starttag(self, tag, attrs):
            if tag == "script":
                self.sources.extend(value for key, value in attrs if key == "src")

    destination = tmp_path / "smoke.html"
    go.Figure(go.Scatter(x=[0, 1], y=[0, 1])).write_html(
        destination, include_plotlyjs=True, full_html=True, auto_open=False
    )
    html = destination.read_text(encoding="utf-8")
    parser = Scripts()
    parser.feed(html)
    assert "plotly.js" in html
    assert "Plotly.newPlot" in html
    assert not parser.sources
    assert destination.resolve().is_relative_to(ROOT / ".tmp")


def test_rich_terminal_output():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    output = StringIO()
    console = Console(file=output, force_terminal=True, width=80)
    table = Table("Check", "Status")
    table.add_row("M0 environment", "PASS")
    console.print(Panel(table, title="Environment smoke test"))
    rendered = output.getvalue()
    assert "M0 environment" in rendered
    assert "PASS" in rendered
    print(rendered)


def test_local_cache_and_environment_paths(tmp_path):
    settings = runpy.run_path(str(ROOT / "scripts/project.py"))
    for name, relative in settings["LOCAL_PATHS"].items():
        assert Path(os.environ[name]).resolve() == ROOT / relative, name
        assert Path(os.environ[name]).resolve().is_relative_to(ROOT), name
    assert Path(tempfile.gettempdir()).resolve() == ROOT / ".tmp"
    assert tmp_path.resolve().is_relative_to(ROOT / ".tmp")
    assert Path(sys.pycache_prefix).resolve() == ROOT / ".cache/pycache"
    assert os.environ["UV_PYTHON_INSTALL_REGISTRY"] == "0"
    assert os.environ["UV_PYTHON_NO_REGISTRY"] == "1"
    assert os.environ["UV_NO_MODIFY_PATH"] == "1"
    assert sys.flags.no_user_site == 1
    for args, expected in [
        (["cache", "dir"], ROOT / ".uv-cache"),
        (["python", "dir"], ROOT / ".tools/python"),
    ]:
        actual = subprocess.check_output([str(settings["UV"]), *args], text=True).strip()
        assert Path(actual).resolve() == expected


def test_uv_version():
    settings = runpy.run_path(str(ROOT / "scripts/project.py"))
    actual = subprocess.check_output([str(settings["UV"]), "--version"], text=True)
    assert actual.split()[1] == settings["UV_VERSION"]


def test_environment_evidence(tmp_path):
    """Leave real machine/version evidence in ignored repository-local output."""
    import torch

    evidence = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
        "packages": {name: version(name) for name in PACKAGES},
        "temp": tempfile.gettempdir(),
    }
    destination = tmp_path / "environment.json"
    destination.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
