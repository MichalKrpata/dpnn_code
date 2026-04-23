import argparse

import matplotlib.pyplot as plt
import torch

from simulators.single_pendulum import SinglePendulum
from simulators.double_pendulum import DoublePendulum
from trajectory_dataset import create_dataset_from_simulator, create_dataloaders
from energy import EnergyNet
from L_builders import CanonicalL, LinearL, NeuralL, TrainableConstantL
from train import TrainableModel, train, train_and_simulate


def create_simulator(args, device):
    """Creates the system based on args"""
    if args.system == "single":
        return SinglePendulum(m=args.m1, l=args.l1, g=args.g, device=device).to(device)
    else:
        return DoublePendulum(m1=args.m1, m2=args.m2, l1=args.l1, l2=args.l2, g=args.g, device=device).to(device)


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
    ax.plot(epochs, history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_trajectory(simulator, model, args, device):
    """Plots angle(s) over time for true vs predicted rollout"""
    z0 = simulator.random_initial_conditions(n_traj=1)
    _, z_true, _ = simulator.simulate_batch(z0, args.dt, args.eval_steps, method="rk4")
    _, z_pred, _ = model.simulate_batch(z0, args.dt, args.eval_steps, method="rk4")

    z_true = z_true[0].detach().cpu()
    z_pred = z_pred[0].detach().cpu()
    t = torch.arange(z_true.shape[0]) * args.dt

    n_angles = 1 if args.system == "single" else 2
    fig, axes = plt.subplots(n_angles, 1, figsize=(9, 3 * n_angles), sharex=True)
    if n_angles == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, z_true[:, i], color="#606060", lw=1.8, ls="--", label="True")
        ax.plot(t, z_pred[:, i], color="#3173F7", lw=1.2, label="Predicted")
        ax.set_ylabel(f"θ{i+1}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"{args.system.title()} pendulum — {args.eval_steps} steps")
    plt.tight_layout()
    plt.show()


def build_parser():
    p = argparse.ArgumentParser(
        description="Train a learned Hamiltonian on a pendulum system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # system parameters
    p.add_argument("--system", default="single", choices=["single", "double"])
    p.add_argument("--m1", type=float, default=1.0, help="Mass of bob 1")
    p.add_argument("--m2", type=float, default=1.0, help="Mass of bob 2 (double only)")
    p.add_argument("--l1", type=float, default=1.0, help="Length of rod 1")
    p.add_argument("--l2", type=float, default=1.0, help="Length of rod 2 (double only)")
    p.add_argument("--g", type=float, default=9.81, help="Gravitational acceleration")

    # model parameters
    p.add_argument("--L_type", default="neural_net", choices=["canonical", "trainable_constant", "linear", "neural_net"])
    p.add_argument("--hidden_dim_H", type=int, default=128)
    p.add_argument("--hidden_dim_L", type=int, default=128)
    p.add_argument("--layers_H", type=int, default=2)
    p.add_argument("--layers_L", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)

    # training parameters
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--n_points", type=int, default=10, help="Points per trajectory (#transitions = n_points - 1)")
    p.add_argument("--n_trajectories_train", type=int, default=200)
    p.add_argument("--n_trajectories_val", type=int, default=40)
    p.add_argument("--jacobi_loss", type=float, default=-1.0)
    p.add_argument("--loss_method", default="exact forward",
                   choices=["random", "exact forward", "exact backward", "spectral", "batch_max", "random loop", "exact_manual"])
    p.add_argument("--sim_batch", action="store_true", help="Simulate fresh data each batch instead of a fixed dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", action="store_true")

    # plotting parameters
    p.add_argument("--plot_error", action="store_true", help="Plot loss curves after training")
    p.add_argument("--plot_trajectory", action="store_true", help="Plot a rollout after training")
    p.add_argument("--eval_steps", type=int, default=256)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"\nSystem  : {args.system} pendulum")
    print(f"Device  : {device}")
    print(f"Epochs  : {args.epochs}  |  batch={args.batch_size}  |  lr={args.lr}")
    print(f"dt={args.dt}  n_points={args.n_points}  total_T={args.dt * (args.n_points - 1):.3f}\n")

    simulator = create_simulator(args, device)
    model = create_model(args, simulator.dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.sim_batch:
        print("Training with live simulation …\n")
        history = train_and_simulate(
            model, simulator, args.n_trajectories_train * (args.n_points - 1), args.batch_size,
            optimizer, args.dt, args.epochs, device=device,
            jacobi_loss=args.jacobi_loss, loss_method=args.loss_method, loss_iter=1,
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
            jacobi_loss=args.jacobi_loss, loss_method=args.loss_method,
        )

    print(f"\nFinal train loss : {history['train_loss'][-1]:.4e}")
    print(f"Final val   loss : {history['val_loss'][-1]:.4e}\n")

    if args.plot_error:
        plot_training(history)

    if args.plot_trajectory:
        plot_trajectory(simulator, model, args, device)