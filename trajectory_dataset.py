import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class TrajectoryDataset(Dataset):
    """Dataset containing pairs of consecutive states"""

    def __init__(self, states: torch.Tensor, next_states: torch.Tensor, 
                device: Optional[torch.device] = None):
        """
        states: (n_trajectories, n_points, state_dim) or (n_samples, state_dim)
        next_states: (n_trajectories, n_points, state_dim) or (n_samples, state_dim)
        """
        super().__init__()

        assert states.shape == next_states.shape, \
            f"States and next_states must have same shape, got {states.shape} and {next_states.shape}"
        
        if states.ndim == 3:
            self.n_trajectories, self.n_points, self.state_dim = states.shape
            self.total_samples = self.n_trajectories * self.n_points

            states = states.reshape(-1, self.state_dim)
            next_states = next_states.reshape(-1, self.state_dim)
        else:
            # the data is flattened
            self.total_samples = states.shape[0]
            self.state_dim = states.shape[1]
            self.n_trajectories = None
            self.n_points = None
        
        if device is None:
            # determine size of the dataset to decide, whether to store on the GPU
            size_gb = states.numel() * states.element_size() * 2 / 1e9

            keep_on_gpu = (size_gb < 1.0) and torch.cuda.is_available()

            device = torch.device('cuda' if (keep_on_gpu and torch.cuda.is_available()) else 'cpu')
        
        self.states = states.to(device)
        self.next_states = next_states.to(device)
        self.device = device
        
        

    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.next_states[idx]
    
    def to(self, device: torch.device) -> 'TrajectoryDataset':
        """Move entire dataset to device"""
        self.states = self.states.to(device)
        self.next_states = self.next_states.to(device)
        self.device = device
        return self
    
    def get_trajectory(self, traj_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a full trajectory"""
        if self.n_trajectories is None:
            raise ValueError("Dataset was not created from 3D trajectory data")
        
        start_idx = traj_idx * self.n_points
        end_idx = start_idx + self.n_points
        
        return self.states[start_idx:end_idx], self.next_states[start_idx:end_idx]
    

def create_dataset_from_simulator(simulator, n_trajectories, dt, n_steps, seed=None, initial_states=None, method='rk4'):
    """Receives a simulator and creates a TrajectoryDataset by simulating a given number of trajectories"""
    if initial_states is None:
        initial_states = simulator.random_initial_conditions(n_trajectories, seed=seed)
    
    _, z_traj, _ = simulator.simulate_batch(initial_states, dt, n_steps, method=method)
    
    states = z_traj[:, :-1, :]
    next_states = z_traj[:, 1:, :]
    
    dataset = TrajectoryDataset(states, next_states)
    return dataset


def create_dataloaders(
    train_dataset: TrajectoryDataset,
    val_dataset: Optional[TrajectoryDataset] = None,
    batch_size: int = 64,
    shuffle_train: bool = True,
    num_workers: int = 0,
    device: Optional[torch.device] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Creates a torch.utils.data.Dataloader from two datasets"""
    if device is not None and isinstance(device, str):
        device = torch.device(device)
        
    pin_memory = (train_dataset.device.type != 'cuda') and (device is not None and device.type == 'cuda')
    
    if train_dataset.device.type == 'cuda' and num_workers > 0:
        print("Warning: Dataset is on GPU. Forcing num_workers=0 to prevent CUDA multiprocessing crash.")
        num_workers = 0
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader
