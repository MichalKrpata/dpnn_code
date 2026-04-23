import pytest
import subprocess

potentials = ["harmonic", "gravity", "lennard"]
L_types = ["neural_net"]
loss_methods = ["random", "exact forward", "exact backward", "spectral",
                "random loop", "exact_manual", "batch_max", None]

@pytest.mark.parametrize("potential", potentials)
@pytest.mark.parametrize("L_type", L_types)
@pytest.mark.parametrize("loss_method", loss_methods)
@pytest.mark.parametrize("sim_batch", [True, False])
@pytest.mark.parametrize("cuda", [True, False])
def test_runs(potential, L_type, loss_method, sim_batch, cuda):
    if not loss_method:
        loss_method = "exact forward"
        jacobi_loss = 0.0
    else:
        jacobi_loss = 0.01
        
    cmd = [
        "python", "-m", "examples.particle_example",
        "--potential", potential,
        "--L_type", L_type,
        "--loss_method", loss_method,
        "--jacobi_loss", str(jacobi_loss),
        "--epochs", "1",
        "--n_points", "3",
        "--batch_size", "4",

        "--n_particles", "2",
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
