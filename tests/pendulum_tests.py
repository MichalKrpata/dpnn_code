import pytest
import subprocess

systems = ["single", "double"]
L_types = ["neural_net", "canonical"]
loss_methods = ["random", "exact forward"]

@pytest.mark.parametrize("system", systems)
@pytest.mark.parametrize("L_type", L_types)
@pytest.mark.parametrize("loss_method", loss_methods)
@pytest.mark.parametrize("sim_batch", [True, False])
@pytest.mark.parametrize("cuda", [True, False])
def test_pendulum_runs(system, L_type, loss_method, sim_batch, cuda):
    cmd = [
        "python", "-m", "examples.pendulum_example",
        "--system", system,
        "--L_type", L_type,
        "--loss_method", loss_method,
        "--jacobi_loss", "0.01",

        "--epochs", "1",
        "--n_points", "2",
        "--batch_size", "2",
        "--n_trajectories_train", "5",
        "--n_trajectories_val", "2",
        "--hidden_dim_H", "16",
        "--hidden_dim_L", "16",
        "--layers_H", "1",
        "--layers_L", "1",
    ]

    if sim_batch:
        cmd.append("--sim_batch")
    if cuda:
        cmd.append("--cuda")

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr