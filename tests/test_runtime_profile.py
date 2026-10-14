import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("override, expected", [(None, "100"), ("50", "50"), ("100", "100")])
def test_ops_profile_is_set_before_mlx_import(override, expected):
    env = dict(os.environ)
    env.pop("MLX_MAX_OPS_PER_BUFFER", None)
    if override is not None:
        env["MLX_MAX_OPS_PER_BUFFER"] = override
    code = "import exlr8, os, sys\nassert 'mlx.core' not in sys.modules\nprint(os.environ['MLX_MAX_OPS_PER_BUFFER'])\n"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("flags, expected", [
    (["--resident"], 440),
    (["--resident", "--no-optimize"], 416),
    (["--resident", "--no-optimize", "--memory-limit-gib", "384"], 384),
    ([], 96),
])
def test_memory_defaults_fit_the_selected_expert_format(monkeypatch, flags, expected):
    from exlr8 import cli

    received = []
    monkeypatch.setattr(cli, "generate", lambda args: received.append(args.memory_limit_gib))
    monkeypatch.setattr(sys, "argv", ["exlr8", "generate", "--prompt", "test", *flags])
    cli.main()
    assert received == [expected]


def test_native_resident_memory_minimum(monkeypatch, capsys):
    from exlr8 import cli

    monkeypatch.setattr(sys, "argv", ["exlr8", "generate", "--prompt", "test", "--resident",
                                     "--memory-limit-gib", "384"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "at least 416 GiB" in capsys.readouterr().err
