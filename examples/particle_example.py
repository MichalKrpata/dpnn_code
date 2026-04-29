import argparse
 
import matplotlib.pyplot as plt
import torch
 
from simulators.particle_system import (
    GravitationalPotential,
    HarmonicPotential,
    LennardJonesPotential,
    ParticleSystem,
)
from trajectory_dataset import create_dataset_from_simulator, create_dataloaders
from energy import EnergyNet
from L_builders import CanonicalL, LinearL, NeuralL, TrainableConstantL
from train import TrainableModel, train, train_and_simulate
 

 
def create_simulator(args, device):
    """Creates the system based on args"""
    potential_map = {
        "harmonic": lambda: HarmonicPotential(n_particles=args.n_particles, k=args.spring_k),
        "gravity": lambda: GravitationalPotential(n_particles=args.n_particles, G=args.gravity_G),
        "lennard": lambda: LennardJonesPotential(epsilon=args.lj_epsilon, sigma=args.lj_sigma),
    }
    return ParticleSystem(
        n_particles=args.n_particles,
        dim_spatial=args.dim_spatial,
        potential_fn=potential_map[args.potential](),
        device=device,
    ).to(device)
 
 
def create_model(args, dim, device):
    """Creates the DPNN model based on args"""
    L_map = {
        "canonical": lambda: CanonicalL(dim),
        "trainable_constant": lambda: TrainableConstantL(dim),
        "linear": lambda: LinearL(dim),
        "neural_net": lambda: NeuralL(dim, hidden_dim=args.hidden_dim_L, layers=args.layers_L, dropout=args.dropout),
    }
    L_net = L_map[args.L_type]()
    H_net = EnergyNet(dim, hidden_dim=args.hidden_dim_H, layers=args.layers_H, dropout=args.dropout)
    return TrainableModel(hamiltonian=H_net, L_matrix=L_net).to(device)
 
 
def plot_training(history):
    """Plots the loss curves"""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train")
    ax.plot(epochs, history["val_loss"],   label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
 
 
def plot_trajectory(simulator, model, args, device):
    """Plots true and simulated trajectories based on parameters from args"""
    z0 = simulator.random_initial_conditions(n_traj=1)
    _, z_true, _ = simulator.simulate_batch(z0, args.dt, args.eval_steps, method="rk4")
    _, z_pred, _ = model.simulate_batch(z0, args.dt, args.eval_steps, method="rk4")
 
    q_dim = args.n_particles * args.dim_spatial
    q_true = z_true[0, :, :q_dim].reshape(-1, args.n_particles, args.dim_spatial).detach().cpu().numpy()
    q_pred = z_pred[0, :, :q_dim].reshape(-1, args.n_particles, args.dim_spatial).detach().cpu().numpy()
 
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(args.n_particles):
        ax.plot(q_true[:, i, 0], q_true[:, i, 1], color="#606060", lw=1.8, ls="--", label="True" if i == 0 else "")
        ax.plot(q_pred[:, i, 0], q_pred[:, i, 1], color="#3173F7", lw=1.2, label="Predicted" if i == 0 else "")
        ax.scatter(q_pred[0, i, 0], q_pred[0, i, 1], color="black", s=14, zorder=5)
 
    ax.set_title(f"{args.potential.title()} — N={args.n_particles}, {args.eval_steps} steps")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
 
 
def build_parser():
    p = argparse.ArgumentParser(
        description="Train a learned Hamiltonian on a particle system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    # system parameters
    p.add_argument("--potential", default="harmonic", choices=["harmonic", "gravity", "lennard"])
    p.add_argument("--n_particles", type=int, default=10)
    p.add_argument("--dim_spatial", type=int, default=2)
    p.add_argument("--spring_k", type=float, default=1.0,  help="Spring constant (harmonic)")
    p.add_argument("--gravity_G", type=float, default=1e-3, help="Gravitational constant (gravity)")
    p.add_argument("--lj_epsilon", type=float, default=0.19, help="Lennard-Jones ε (lennard)")
    p.add_argument("--lj_sigma", type=float, default=3.19, help="Lennard-Jones σ (lennard)")
 
    # model parameters
    p.add_argument("--L_type", default="neural_net",choices=["canonical", "trainable_constant", "linear", "neural_net"])
    p.add_argument("--hidden_dim_H", type=int, default=512)
    p.add_argument("--hidden_dim_L", type=int, default=128)
    p.add_argument("--layers_H", type=int, default=3)
    p.add_argument("--layers_L", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
 
    # training parameters
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--n_points", type=int, default=2, help="Number of points, i.e. #transitions = n_points - 1")
    p.add_argument("--n_trajectories_train", type=int, default=4096)
    p.add_argument("--n_trajectories_val", type=int, default=128)
    p.add_argument("--jacobi_loss", type=float, default=0.002, help="Jacobi loss weight, set positive to use it.")
    p.add_argument("--loss_method", default="exact forward",
                   choices=["random", "exact forward", "exact_forward", "exact backward", "exact_backward", "spectral", 
                            "batch_max", "monte_carlo", "random loop", "random_loop", "Monte Carlo"])
    p.add_argument("--scheme", default="CN", choices=["CN", "IMR", "RK4"])
    p.add_argument("--sim_batch", default=False, action="store_true", help="Simulate fresh data for each batch instead of a fixed dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", action="store_true")
 
    # plotting parameters
    p.add_argument("--plot_error", action="store_true", help="Plot loss curves after training")
    p.add_argument("--plot_trajectory", action="store_true", help="Plot a rollout after training")
    p.add_argument("--eval_steps", type=int, default=256, help="Rollout length for --plot_trajectory")
 
    return p
 
 
if __name__ == "__main__":
    args = build_parser().parse_args()
 
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
 
    print(f"\nPotential  : {args.potential}")
    print(f"Particles  : {args.n_particles}  dim={args.dim_spatial}")
    print(f"Device     : {device}")
    print(f"Epochs     : {args.epochs}  |  batch={args.batch_size}  |  lr={args.lr}")
    print(f"dt={args.dt}  n_steps={args.n_points}  total_T={args.dt * (args.n_points - 1):.3f}\n")
 
    simulator = create_simulator(args, device)
    model = create_model(args, simulator.dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
 
    if args.sim_batch:
        print("Training with live simulation …\n")
        history = train_and_simulate(
            model, simulator, args.n_trajectories_train * (args.n_points - 1), args.batch_size, 
            optimizer, args.dt, args.epochs, device=device, jacobi_loss=args.jacobi_loss, 
            loss_method=args.loss_method, loss_iter=1, scheme=args.scheme
        )

    else:
        print("Generating datasets …")
        train_dataset = create_dataset_from_simulator(
            simulator, n_trajectories=args.n_trajectories_train,
            dt=args.dt, n_steps=args.n_points, seed=args.seed,
        )
        val_dataset = create_dataset_from_simulator(
            simulator, n_trajectories=args.n_trajectories_val,
            dt=args.dt, n_steps=args.n_points, seed=args.seed + 100,
        )
        print(f"  train: {len(train_dataset)} samples  |  val: {len(val_dataset)} samples\n")
 
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, batch_size=args.batch_size, device=device,
        )
 
        print("Training …\n")
        history = train(
            model, train_loader, val_loader, optimizer, args.dt, args.epochs, device=device,
            jacobi_loss=args.jacobi_loss, loss_method=args.loss_method, scheme=args.scheme
        )
 
    print(f"\nFinal train loss : {history['train_loss'][-1]:.4e}")
    print(f"Final val   loss : {history['val_loss'][-1]:.4e}\n")
 
    if args.plot_error:
        plot_training(history)
 
    if args.plot_trajectory:
        plot_trajectory(simulator, model, args, device)
