# Direct Poisson Neural Networks (DPNN)

This repository contains the Python implementation for my thesis on **Direct Poisson Neural Networks**. The framework is designed to learn the dynamics of physical systems while preserving their underlying Poisson structure.

## Project Structure

- **Root Directory**: Contains core logic, including ML models (Hamiltonian and Poisson Matrix architectures), training loops, and utility scripts.
- **`simulators/`**: Contains simulator classes and physics engines for the different systems.
- **`examples/`**: A folder containing 5 standalone scripts for different physical systems.

## Installation

1. Clone the repository.
2. Install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Running the Examples

The examples contain the entire pipeline: model creation, data simulation, training, and evaluation. Because the examples import logic from the core modules in the root directory, they must be run as **modules** using the `-m` flag.

### Basic Usage
To run an example with its default settings, use the following commands from the root directory:

```bash
python -m examples.particle_example
python -m examples.orbit_example
python -m examples.flywheel_example
python -m examples.crb_example
python -m examples.pendulum_example
```

### Advanced Usage with Arguments
You can customize the physics, the model architecture, and the training loop via command-line arguments.

**Example: Training a particle system with the Lennard-Jones potential and 20 particles:**
```bash
python -m examples.particle_example --potential lennard --n_particles 20 --cuda --plot_trajectory
```

### General Argument Overview

| Category | Key Arguments |
| :--- | :--- |
| **Model** | `--L_type`, `--hidden_dim_H`, `--hidden_dim_L`, `--layers_H`, `--layers_L`, `--dropout` |
| **Training** | `--epochs`, `--batch_size`, `--lr`, `--loss_method`, `--jacobi_loss` (weight) |
| **Data** | `--dt`, `--n_points` (points per trajectory), `--n_trajectories_train`, `--sim_batch` |
| **Plotting** | `--plot_error`, `--plot_trajectory`, `--eval_steps` |

For a full list of flags and descriptions for a specific system, run:
```bash
python -m examples.<filename> --help
```
