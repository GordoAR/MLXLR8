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
