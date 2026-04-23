import argparse
 
import matplotlib.pyplot as plt
import torch
 
from simulators.particle_system import GravitationalPotential, ParticleSystem
from trajectory_dataset import create_dataset_from_simulator, create_dataloaders
from energy import EnergyNet
from L_builders import CanonicalL, LinearL, NeuralL, TrainableConstantL
from train import TrainableModel, train
 
 
def make_initial_conditions(n_traj, n_planets, radii, masses, G, device, seed=None):
    """
    Builds a batch of orbital initial conditions. The sun is particle 0,
    planets are placed on slightly randomised circular orbits around it.
    """
    if seed is not None:
        torch.manual_seed(seed)
 
    n_particles = n_planets + 1
    dim_spatial = 2
    M_sun = masses[0].item()
 
    q0 = torch.zeros((n_traj, n_particles, dim_spatial), device=device)
    p0 = torch.zeros((n_traj, n_particles, dim_spatial), device=device)
 
    for i, r_mean in enumerate(radii):
        phi = torch.rand(n_traj, device=device) * 2 * torch.pi
        r = r_mean * (1 + (torch.rand(n_traj, device=device) - 0.5) * 0.04)
        m = masses[i + 1].item()
 
        q0[:, i + 1, 0] = r * torch.cos(phi)
        q0[:, i + 1, 1] = r * torch.sin(phi)
 
        v_mag = (G * M_sun / r) ** 0.5
        p0[:, i + 1, 0] = -m * v_mag * torch.sin(phi)
        p0[:, i + 1, 1] =  m * v_mag * torch.cos(phi)
 
    # Sun gets the negative total momentum to conserve it
    p0[:, 0, :] = -torch.sum(p0[0, 1:, :], dim=0)
 
    return q0, p0
 
 
def create_simulator(args, masses, device):
    """Creates the system based on args"""
    n_particles = args.n_planets + 1
    potential = GravitationalPotential(n_particles, G=args.G, m=masses)
    return ParticleSystem(
        n_particles=n_particles,
        dim_spatial=2,
        potential_fn=potential,
        m=masses,
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
    ax.plot(epochs, history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
 
 
def plot_trajectory(system, model, args, z_val, device):
    """Plots true and simulated trajectories based on parameters from args"""
    n_particles = args.n_planets + 1
    colors = ["black", "#FFA530", "#FF313F", "#AD1BD2", "#3173F7",
              "#2CA02C", "#8C564B", "#E377C2", "#7F7F7F"]
 
    _, z_traj, _ = model.simulate_batch(z_val[:1], args.dt, args.eval_steps, method="rk4")
 
    traj = z_traj[0].detach().cpu().numpy().reshape(-1, 2, n_particles, 2)
    positions = traj[:, 0]
 
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(n_particles):
        color = colors[i % len(colors)]
        x, y = positions[:, i, 0], positions[:, i, 1]
        s = 2 if i == 0 else 1
        ax.plot(x, y, color=color, alpha=0.9, linewidth=1.2)
        ax.scatter(x[0], y[0], marker="o", s=50*s, color=color, edgecolors="black", linewidths=0.6, zorder=10)
        ax.scatter(x[-1], y[-1], marker="X", s=80*s, color=color, edgecolors="black", linewidths=0.6, zorder=10)
 
    ax.set_title(f"Gravitational Trajectories — {n_particles} bodies, {args.eval_steps} steps")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()
 
 
def build_parser():
    p = argparse.ArgumentParser(
        description="Train a learned Hamiltonian on a gravitational particle system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    # system parameters
    p.add_argument("--n_planets", type=int, default=2,
                   help="Number of planets (total particles = n_planets + 1 sun)")
    p.add_argument("--radii", type=float, nargs="+", default=[1.5, 2.5],
                   help="Mean orbital radius of each planet")
    p.add_argument("--sun_mass", type=float, default=100.0)
    p.add_argument("--planet_mass", type=float, default=0.001,
                   help="Mass for all planets (shared)")
    p.add_argument("--G", type=float, default=1.0, help="Gravitational constant")
 
    # model parameters
    p.add_argument("--L_type", default="neural_net", choices=["canonical", "trainable_constant", "linear", "neural_net"])
    p.add_argument("--hidden_dim_H", type=int, default=128)
    p.add_argument("--hidden_dim_L", type=int, default=512)
    p.add_argument("--layers_H", type=int, default=2)
    p.add_argument("--layers_L", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
 
    # training parameters
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--n_points", type=int, default=17, help="Points per trajectory (#transitions = n_points - 1)")
    p.add_argument("--n_traj_train", type=int, default=512)
    p.add_argument("--n_traj_val", type=int, default=64)
    p.add_argument("--jacobi_loss", type=float, default=0.001)
    p.add_argument("--loss_method", default="exact forward",
                   choices=["random", "exact forward", "exact backward", "spectral", "batch_max", "random loop", "exact_manual"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", action="store_true")
 
    # plotting parameters
    p.add_argument("--plot_error", action="store_true", help="Plot loss curves after training")
    p.add_argument("--plot_trajectory", action="store_true", help="Plot a rollout after training")
    p.add_argument("--eval_steps", type=int, default=2000)
 
    return p
 
 
if __name__ == "__main__":
    args = build_parser().parse_args()
 
    if len(args.radii) != args.n_planets:
        raise ValueError(f"--radii must have exactly n_planets={args.n_planets} values, got {len(args.radii)}")
 
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
 
    masses = torch.tensor(
        [args.sun_mass] + [args.planet_mass] * args.n_planets,
        device=device
    )
 
    print(f"\nBodies    : {args.n_planets + 1} ({args.n_planets} planets + sun)")
    print(f"Radii     : {args.radii}")
    print(f"Device    : {device}")
    print(f"Epochs    : {args.epochs}  |  batch={args.batch_size}  |  lr={args.lr}")
    print(f"dt={args.dt}  n_points={args.n_points}  total_T={args.dt * (args.n_points - 1):.3f}\n")
 
    simulator = create_simulator(args, masses, device)
    model = create_model(args, simulator.dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
 
    print("Generating datasets …")
    q0_train, p0_train = make_initial_conditions(
        args.n_traj_train, args.n_planets, args.radii, masses, args.G, device, seed=args.seed
    )
    z0_train = simulator._flatten_qp(q0_train, p0_train)
 
    q0_val, p0_val = make_initial_conditions(
        args.n_traj_val, args.n_planets, args.radii, masses, args.G, device, seed=args.seed + 100
    )
    z0_val = simulator._flatten_qp(q0_val, p0_val)
 
    train_dataset = create_dataset_from_simulator(
        simulator, n_trajectories=0, dt=args.dt, n_steps=args.n_points, initial_states=z0_train
    )
    val_dataset = create_dataset_from_simulator(
        simulator, n_trajectories=0, dt=args.dt, n_steps=args.n_points, initial_states=z0_val
    )
    print(f"  train: {len(train_dataset)} samples  |  val: {len(val_dataset)} samples\n")
 
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, batch_size=args.batch_size, device=device
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
        plot_trajectory(simulator, model, args, z0_val, device)