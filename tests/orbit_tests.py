import pytest
import subprocess

L_types = ["neural_net", "canonical"]
loss_methods = ["random", "exact forward", None]

@pytest.mark.parametrize("L_type", L_types)
@pytest.mark.parametrize("loss_method", loss_methods)
@pytest.mark.parametrize("n_planets", [1, 2])
@pytest.mark.parametrize("cuda", [True, False])
def test_orbit_runs(L_type, loss_method, n_planets, cuda):
    radii = ["1.5"] if n_planets == 1 else ["1.5", "2.5"]

    if not loss_method:
        loss_method = "exact forward"
        jacobi_loss = 0.0
    else:
        jacobi_loss = 0.01

    cmd = [
        "python", "-m", "examples.orbit_example",

        "--n_planets", str(n_planets),
        "--radii", *radii,
        "--L_type", L_type,
        "--loss_method", loss_method,
        "--jacobi_loss", str(jacobi_loss),

        "--epochs", "1",
        "--n_points", "2",
        "--batch_size", "2",
        "--n_traj_train", "5",
        "--n_traj_val", "2",
        "--hidden_dim_H", "16",
        "--hidden_dim_L", "16",
        "--layers_H", "1",
        "--layers_L", "1",
    ]

    if cuda:
        cmd.append("--cuda")

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr