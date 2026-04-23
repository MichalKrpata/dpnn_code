import argparse

import matplotlib.pyplot as plt
import torch

from simulators.coupled_rigid_bodies import CoupledRigidBodies3D
from trajectory_dataset import create_dataset_from_simulator, create_dataloaders
from energy import EnergyNet
from L_builders import CanonicalL, LinearL, NeuralL, TrainableConstantL
from train import TrainableModel, train, train_and_simulate


def create_simulator(args, device):
    """Creates the system based on args"""
    return CoupledRigidBodies3D(
        m1=args.m1, m2=args.m2,
        J1_diag=args.J1, J2_diag=args.J2,
        S1_0=args.S1, S2_0=args.S2,
        device=device,
    )


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
    """Plots true and simulated trajectories based on parameters from args"""
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
    """Plots angular momenta (Pi1, Pi2) and Euler angles for true vs predicted trajectory"""
    z0 = simulator.random_initial_conditions(n_traj=1)
    _, z_true, _ = simulator.simulate_batch(z0, args.dt, args.eval_steps, method="rk4")
    _, z_pred, _ = model.simulate_batch(z0, args.dt, args.eval_steps, method="rk4")

    z_true = z_true[0].detach().cpu()
    z_pred = z_pred[0].detach().cpu()
    t = torch.arange(z_true.shape[0]) * args.dt

    labels = [
        ("Π₁", ["x", "y", "z"], slice(0, 3)),
        ("Π₂", ["x", "y", "z"], slice(3, 6)),
        ("Angles", ["phi", "theta", "psi"], slice(6, 9)),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    colors = ["#3173F7", "#FF313F", "#FFA530"]

    for ax, (title, comp_labels, sl) in zip(axes, labels):
        for j, (comp, color) in enumerate(zip(comp_labels, colors)):
            idx = sl.start + j
            ax.plot(t, z_true[:, idx], color=color, lw=1.8, ls="--",
                    label=f"{comp} true")
            ax.plot(t, z_pred[:, idx], color=color, lw=1.2,
                    label=f"{comp} pred")
        ax.set_ylabel(title)
        ax.legend(ncol=3, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Coupled rigid bodies — {args.eval_steps} steps")
    plt.tight_layout()
    plt.show()


def build_parser():
    p = argparse.ArgumentParser(
        description="Train a learned Hamiltonian on two coupled rigid bodies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # system parameters
    p.add_argument("--m1", type=float, default=1.0, help="Mass of body 1")
    p.add_argument("--m2", type=float, default=1.0, help="Mass of body 2")
    p.add_argument("--J1", type=float, nargs=3, default=[1.0, 2.0, 3.0],
                   help="Moments of inertia for body 1")
    p.add_argument("--J2", type=float, nargs=3, default=[2.0, 1.0, 3.0],
                   help="Moments of inertia for body 2")
    p.add_argument("--S1", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                   help="Offset vector for body 1")
    p.add_argument("--S2", type=float, nargs=3, default=[0.0, 0.0, -1.0],
                   help="Offset vector for body 2")

    # model parameters
    p.add_argument("--L_type", default="neural_net", choices=["canonical", "trainable_constant", "linear", "neural_net"])
    p.add_argument("--hidden_dim_H", type=int, default=256)
    p.add_argument("--hidden_dim_L", type=int, default=128)
    p.add_argument("--layers_H", type=int, default=3)
    p.add_argument("--layers_L", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)

    # training parameters
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--n_points", type=int, default=17, help="Points per trajectory (#transitions = n_points - 1)")
    p.add_argument("--n_trajectories_train", type=int, default=2048)
    p.add_argument("--n_trajectories_val", type=int, default=40)
    p.add_argument("--jacobi_loss", type=float, default=0.005)
    p.add_argument("--loss_method", default="random",
                   choices=["random", "exact forward", "exact backward", "spectral", "batch_max", "random loop", "exact_manual"])
    p.add_argument("--sim_batch", action="store_true", help="Simulate fresh data each batch instead of a fixed dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", action="store_true")

    # plotting parameters
    p.add_argument("--plot_error", action="store_true", help="Plot loss curves after training")
    p.add_argument("--plot_trajectory", action="store_true", help="Plot a rollout after training")
    p.add_argument("--eval_steps", type=int, default=128)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"\nSystem    : coupled rigid bodies")
    print(f"m1={args.m1}  m2={args.m2}")
    print(f"J1={args.J1}  J2={args.J2}")
    print(f"S1={args.S1}  S2={args.S2}")
    print(f"Device    : {device}")
    print(f"Epochs    : {args.epochs}  |  batch={args.batch_size}  |  lr={args.lr}")
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