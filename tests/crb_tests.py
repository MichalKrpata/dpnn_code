import pytest
import subprocess

L_types = ["neural_net"]
loss_methods = ["random", None]
scheme = ["CN", "IMR", "RK4"]

@pytest.mark.parametrize("L_type", L_types)
@pytest.mark.parametrize("loss_method", loss_methods)
@pytest.mark.parametrize("sim_batch", [True, False])
@pytest.mark.parametrize("cuda", [True, False])
@pytest.mark.parametrize("scheme", scheme)
def test_rigid_body_runs(L_type, loss_method, sim_batch, cuda, scheme):
    if not loss_method:
        loss_method = "exact forward"
        jacobi_loss = 0.0
    else:
        jacobi_loss = 0.01

    cmd = [
        "python", "-m", "examples.crb_example",

        "--L_type", L_type,
        "--loss_method", loss_method,
        "--jacobi_loss", str(jacobi_loss),
        "--scheme", scheme,

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