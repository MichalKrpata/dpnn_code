import pytest
import subprocess

L_types = ["neural_net", "canonical"]
loss_methods = ["spectral", None]

@pytest.mark.parametrize("L_type", L_types)
@pytest.mark.parametrize("loss_method", loss_methods)
@pytest.mark.parametrize("n_flywheels", [2, 3])
@pytest.mark.parametrize("sim_batch", [True, False])
@pytest.mark.parametrize("cuda", [True, False])
def test_flywheel_runs(L_type, loss_method, n_flywheels, sim_batch, cuda):
    I = ["1.0"] if n_flywheels == 2 else ["1.0", "1.0", "1.0"]
    k = ["1.0"] if n_flywheels == 2 else ["1.0", "1.0"]

    if not loss_method:
        loss_method = "exact forward"
        jacobi_loss = 0.0
    else:
        jacobi_loss = 0.01

    cmd = [
        "python", "-m", "examples.flywheel_example",

        "--n_flywheels", str(n_flywheels),
        "--I", *I,
        "--k", *k,

        "--L_type", L_type,
        "--loss_method", loss_method,
        "--jacobi_loss", str(jacobi_loss),

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

    if cuda:
        cmd.append("--cuda")

    if sim_batch:
        cmd.append("--sim_batch")

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr