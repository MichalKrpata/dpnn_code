import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from typing import Optional
import time

from simulators.base_simulator import BaseSimulator
from L_builders import *


class TrainableModel(BaseSimulator):
    """A model containing the separate components"""
    def __init__(self, 
                 hamiltonian: nn.Module, 
                 L_matrix: nn.Module):
        super().__init__(hamiltonian.dim)
        self.energy = hamiltonian
        self.L_matrix = L_matrix

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the time derivative dz/dt."""
        z_grad = z.detach().requires_grad_(True)
        
        H = self.energy(z_grad)
        
        grad_H = torch.autograd.grad(
            outputs=H.sum(), 
            inputs=z_grad,
            create_graph=self.training
        )[0]
        
        L = self.L_matrix(z) 
        
        grad_H_unsqueezed = grad_H.unsqueeze(-1)
        dz_dt = torch.bmm(L, grad_H_unsqueezed).squeeze(-1)
        
        return dz_dt
    
    def hamiltonian(self, z):
        return self.energy(z)

    def simulate_batch(self, z0, dt, n_steps, method = 'rk4'):
        """Predict the trajectory from the initial condition for the given number of steps"""
        self.eval()
        return super().simulate_batch(z0, dt, n_steps, method)


def movement_loss(model, z_n, z_n1, L_n, dt, scheme='CN'):
    """Calculates the loss from the trajectory"""

    def output(z, L=None):
        H = model.hamiltonian(z)
        grad_H = torch.autograd.grad(
            outputs=H.sum(),
            inputs=z,
            create_graph=model.training
        )[0]
        if L is None:
            L = model.L_matrix(z)
        return torch.bmm(L, grad_H.unsqueeze(-1)).squeeze(-1)
    
    if scheme == 'CN':
        f_n = output(z_n, L_n)
        f_n1 = output(z_n1)
        rhs = 0.5 * (f_n + f_n1)

    elif scheme == 'IMR':
        z_mid = 0.5 * (z_n + z_n1)
        rhs = output(z_mid)

    elif scheme == 'RK4':
        k1 = output(z_n, L_n)
        k2 = output(z_n + 0.5 * dt * k1)
        k3 = output(z_n + 0.5 * dt * k2)
        k4 = output(z_n + dt * k3)
        rhs = (k1 + 2*k2 + 2*k3 + k4) / 6.0

    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    
    lhs = (z_n1 - z_n) / dt

    residual = lhs - rhs
    
    return (residual ** 2).mean()


def energy_loss(model, z_n, z_n1):
    """Calculates the loss from energy conservation"""
    H_n = model.hamiltonian(z_n)
    H_n1 = model.hamiltonian(z_n1)

    return ((H_n1 - H_n)**2).mean()


def jacobi_loss_linear(model):
    """Calculates Jacobi identity loss, only applicable to a linear L matrix"""
    c = model.L_matrix.get_linear_constants()
    dim = c.size(0)

    loss = 0.0
    for m in range(dim):
        A = c[:, :, m]

        term1 = torch.einsum('il,jkl->ijk', A, c)
        term2 = torch.einsum('jl,kil->ijk', A, c)
        term3 = torch.einsum('kl,ijl->ijk', A, c)

        J_m = term1 + term2 + term3
        loss = loss + (J_m * J_m).sum()

    return loss / c.numel()


def jacobi_loss_random(model, z, iter=5, dist="rademacher"):
    B, dim = z.shape
    total_samples = iter * B
    
    z_exp = z.repeat(iter, 1).detach().requires_grad_(True)

    L_exp = model.L_matrix(z_exp)

    # randomly sample a batch of random vectors with the given distribution
    total_samples = iter * B
    if dist == "normal":
        u = torch.randn(total_samples, dim, device=z.device)
        v = torch.randn(total_samples, dim, device=z.device)
        w = torch.randn(total_samples, dim, device=z.device)
    elif dist == "rademacher":
        u = torch.randint(0, 2, (total_samples, dim), device=z.device).float() * 2 - 1
        v = torch.randint(0, 2, (total_samples, dim), device=z.device).float() * 2 - 1
        w = torch.randint(0, 2, (total_samples, dim), device=z.device).float() * 2 - 1
    elif dist == "uniform":
        bound = math.sqrt(3)
        u = torch.rand(total_samples, dim, device=z.device) * 2 * bound - bound
        v = torch.rand(total_samples, dim, device=z.device) * 2 * bound - bound
        w = torch.rand(total_samples, dim, device=z.device) * 2 * bound - bound
    else:
        raise ValueError("dist must be 'normal', 'rademacher' or 'uniform'")

    def compute_term_vec(vec_a, vec_b, vec_c, retain_graph=True):
        # computes one term of the cyclic sum
        S = torch.einsum('bi,bij,bj->b', vec_b, L_exp, vec_c)
        
        La = torch.einsum('bij,bj->bi', L_exp, vec_a)
        
        grad_S = torch.autograd.grad(S.sum(), z_exp, create_graph=True, retain_graph=retain_graph)[0]
        
        return (La * grad_S).sum(dim=1)

    term1 = compute_term_vec(u, v, w, retain_graph=True)
    term2 = compute_term_vec(v, w, u, retain_graph=True)
    term3 = compute_term_vec(w, u, v, retain_graph=True)
    
    loss_i = (term1 + term2 + term3).pow(2)
    
    return loss_i.mean()


def jacobi_loss_batch_max(model, z, L, iter=5):
    """Monte Carlo approximation of Jacobiator's spectral norm"""
    B, dim, _ = L.shape
    
    def compute_term(vec_a, vec_b, vec_c):
        S = torch.einsum('bi,bij,bj->b', vec_b, L, vec_c)
        grad_S = torch.autograd.grad(S.sum(), z, create_graph=True, retain_graph=True)[0]
        La = torch.einsum('bij,bj->bi', L, vec_a)
        return (La * grad_S).sum(dim=1)
    
    worst_loss = torch.tensor(0.0, device=z.device, requires_grad=True)

    # choose the maximum encountered value of the contraction
    for _ in range(iter):
        u = torch.randn(B, dim, device=z.device)
        v = torch.randn(B, dim, device=z.device)
        w = torch.randn(B, dim, device=z.device)

        u = torch.nn.functional.normalize(u, dim=1)
        v = torch.nn.functional.normalize(v, dim=1)
        w = torch.nn.functional.normalize(w, dim=1)

        term1 = compute_term(u, v, w)
        term2 = compute_term(v, w, u)
        term3 = compute_term(w, u, v)
        
        J_violation = (term1 + term2 + term3).pow(2).mean()
        
        if J_violation > worst_loss:
            worst_loss = J_violation

    return worst_loss


def jacobi_loss_exact(model, z, L):
    """Exact Frobenius norm, uses backward AD"""
    def reduced_L(z_in):
        return model.L_matrix(z_in).sum(dim=0)
    
    print(L.shape)

    Lz_grad = torch.autograd.functional.jacobian(reduced_L, z, create_graph=True)
    
    batch_jac = Lz_grad.permute(2, 0, 1, 3)
    
    term1 = torch.einsum('bil,bjkl->bijk', L, batch_jac)
    term2 = term1.permute(0, 2, 3, 1)
    term3 = term1.permute(0, 3, 1, 2)
    
    return (term1 + term2 + term3).pow(2).sum(dim=(1,2,3)).mean()


def jacobi_loss_forward(model, z, L):
    """Exact Frobenius norm, uses forward AD"""
    z_detached = z.detach()
    
    def compute_single_L(z_single):
        return model.L_matrix(z_single.unsqueeze(0)).squeeze(0)

    # forward mode AD on a batch of inputs
    batch_jac = torch.func.vmap(torch.func.jacfwd(compute_single_L))(z_detached)
    
    Lz = model.L_matrix(z)
    
    term1 = torch.einsum('bil,bjkl->bijk', L, batch_jac)
    term2 = term1.permute(0, 2, 3, 1)
    term3 = term1.permute(0, 3, 1, 2)
    
    return (term1 + term2 + term3).pow(2).sum(dim=(1,2,3)).mean()


def jacobi_loss_spectral(model, z, L, iter=5):
    """Iterative approximation of the spectral norm"""
    B, dim, _ = L.shape
    
    u = torch.randn(B, dim, device=z.device)
    v = torch.randn(B, dim, device=z.device)
    w = torch.randn(B, dim, device=z.device)
    
    u = torch.nn.functional.normalize(u, dim=1)
    v = torch.nn.functional.normalize(v, dim=1)
    w = torch.nn.functional.normalize(w, dim=1)

    def get_jacobiator_scalar(u_vec, v_vec, w_vec, create_graph=False):
        # computes the contraction of the Jacobiator

        def cyclic_term(a, b, c):
            S = torch.einsum('bi,bij,bj->b', b, L, c)
            grad_S = torch.autograd.grad(S.sum(), z, create_graph=create_graph, retain_graph=True)[0]
            aL = torch.einsum('bi,bij->bj', a, L)
            return (aL * grad_S).sum(dim=1)
            
        return cyclic_term(u_vec, v_vec, w_vec) + \
               cyclic_term(v_vec, w_vec, u_vec) + \
               cyclic_term(w_vec, u_vec, v_vec)

    # updates the vectors for a given number of times
    i = 0
    while i < iter:
        u.requires_grad_(True)
        J_u = get_jacobiator_scalar(u, v, w, create_graph=True)
        grad_u = torch.autograd.grad(J_u.sum(), u)[0]
        u = torch.nn.functional.normalize(grad_u.detach(), dim=1)
        i += 1
        
        if i < iter:
            v.requires_grad_(True)
            J_v = get_jacobiator_scalar(u, v, w, create_graph=True)
            grad_v = torch.autograd.grad(J_v.sum(), v)[0]
            v = torch.nn.functional.normalize(grad_v.detach(), dim=1)
            i += 1
        
        if i < iter:
            w.requires_grad_(True)
            J_w = get_jacobiator_scalar(u, v, w, create_graph=True)
            grad_w = torch.autograd.grad(J_w.sum(), w)[0]
            w = torch.nn.functional.normalize(grad_w.detach(), dim=1)
            i += 1

    final_violation = get_jacobiator_scalar(u, v, w, create_graph=True)
    
    return (final_violation ** 2).mean()


def jacobi_loss_random_loop(model, z, L, iter=5, dist="rademacher"):
    """Stochastic approximation of the Jacobiator's Frobenius norm (looped)"""
    B, dim = z.shape
    device = z.device
    total_loss = 0.0

    def get_noise(shape):
        if dist == "normal":
            return torch.randn(shape, device=device)
        elif dist == "rademacher":
            return torch.randint(0, 2, shape, device=device).float() * 2 - 1
        elif dist == "uniform":
            bound = math.sqrt(3)
            return torch.rand(shape, device=device) * 2 * bound - bound
        else:
            raise ValueError("dist must be 'normal', 'rademacher' or 'uniform'")

    for _ in range(iter):
        u = get_noise((B, dim))
        v = get_noise((B, dim))
        w = get_noise((B, dim))

        def compute_term_vec(vec_a, vec_b, vec_c):
            S = torch.einsum('bi,bij,bj->b', vec_b, L, vec_c)
            La = torch.einsum('bij,bj->bi', L, vec_a)

            grad_S = torch.autograd.grad(
                S.sum(),
                z,
                create_graph=True,
                retain_graph=True
            )[0]

            return (La * grad_S).sum(dim=1)

        term1 = compute_term_vec(u, v, w)
        term2 = compute_term_vec(v, w, u)
        term3 = compute_term_vec(w, u, v)

        loss_i = (term1 + term2 + term3).pow(2)
        total_loss = total_loss + loss_i.mean()
        
    return total_loss / iter 


def total_loss(model, z_n, z_n1, dt, lambda_jacobi=0.0, lambda_energy=0.0, method="random", iter=5, dist="normal", scheme='CN'):
    """Calculates all used losses"""
    losses = {}

    z_n = z_n.detach().requires_grad_(True)
    z_n1 = z_n1.detach().requires_grad_(True)
    L = model.L_matrix(z_n)
    
    losses['movement'] = movement_loss(model, z_n, z_n1, L, dt, scheme=scheme)

    if lambda_energy > 0.0:
        losses['energy'] = lambda_energy * energy_loss(model, z_n, z_n1)

    if lambda_jacobi > 0.0:
        if isinstance(model.L_matrix, NeuralL):
            is_training = model.training
            model.eval()

            with torch.enable_grad():
                if method == 'random':
                    j_loss = jacobi_loss_random(model, z_n, iter=iter, dist=dist)
                elif method in ['Monte Carlo', 'batch_max']:
                    j_loss = jacobi_loss_batch_max(model, z=z_n, L=L, iter=iter)
                elif method in ["exact backward", "exact_backward"]:
                    j_loss = jacobi_loss_exact(model, z_n, L)
                elif method == "spectral":
                    j_loss = jacobi_loss_spectral(model, z_n, L, iter=iter)
                elif method in ["random loop", "random_loop"]:
                    j_loss = jacobi_loss_random_loop(model, z_n, L, iter=iter)
                elif method in ["exact forward", "exact_forward"]:
                    j_loss = jacobi_loss_forward(model, z_n, L)
                else:
                    raise ValueError(f"Unknown Jacobi loss method: {method}")
            
            if is_training:
                model.train()
            
            losses['jacobi'] = lambda_jacobi * j_loss
            
        elif isinstance(model.L_matrix, LinearL):
            losses['jacobi'] = lambda_jacobi * jacobi_loss_linear(model)
                
    return losses
    

def train(
    model: TrainableModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    dt: float,
    epochs: int,
    device: torch.device,
    max_grad_norm: Optional[float] = None,
    energy_loss: float = 0.0,
    jacobi_loss: float = 0.0,
    loss_method: str = "exact forward",
    loss_iter: int = 5,
    scheme: str = "CN",
    print_every: int = 1
) -> dict:
    """Contains the training and validation loops for a given dataset"""
    history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}

    if isinstance(device, str):
        device = torch.device(device)

    # decays the learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        model.train()
        running_losses = {}
        
        for z_n, z_n1 in train_loader:
            z_n = z_n.to(device)
            z_n1 = z_n1.to(device)
            
            optimizer.zero_grad()
            
            loss_dict = total_loss(model, z_n, z_n1, dt, lambda_jacobi=jacobi_loss, lambda_energy=energy_loss, 
                              method=loss_method, scheme=scheme, iter=loss_iter)
            
            for k, v in loss_dict.items():
                running_losses[f'train_{k}'] = running_losses.get(f'train_{k}', 0.0) + v.item()
                
            total_batch_loss = sum(loss_dict.values())
            total_batch_loss.backward()

            # optionally clip the gradient
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            del loss_dict, total_batch_loss 

        # add all the losses
        current_train_total = 0.0
        for k in running_losses:
            avg_val = running_losses[k] / len(train_loader)
            if k not in history: history[k] = []
            history[k].append(avg_val)
            current_train_total += avg_val
        history['train_loss'].append(current_train_total)
        
        # evaluation loop
        model.eval()
        running_losses_val = {}

        for param in model.parameters():
            param.requires_grad = False
        
        for z_n, z_n1 in val_loader:
            z_n = z_n.to(device)
            z_n1 = z_n1.to(device)
            
            v_loss_dict = total_loss(model, z_n, z_n1, dt, lambda_jacobi=jacobi_loss, lambda_energy=energy_loss, 
                                     method=loss_method, iter=loss_iter, scheme=scheme)
            for k, v in v_loss_dict.items():
                running_losses_val[f'val_{k}'] = running_losses_val.get(f'val_{k}', 0.0) + v.item()

        current_val_total = 0.0
        for k in running_losses_val:
            avg_val = running_losses_val[k] / len(val_loader)
            if k not in history: history[k] = []
            history[k].append(avg_val)
            current_val_total += avg_val
        history['val_loss'].append(current_val_total)

        scheduler.step()

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            #peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            #print("Peak mem: ", peak_mem_mb)
            #torch.cuda.reset_peak_memory_stats(device)

        epoch_time = time.time() - start_time
        history['epoch_times'].append(epoch_time)

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {current_train_total:.3e} | Val: {current_val_total:.3e} | {epoch_time:.1f}s")

        for param in model.parameters():
            param.requires_grad = True

    return history


def train_and_simulate(
    model: TrainableModel,
    simulator: BaseSimulator,
    n_traj_train: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    dt: float,
    epochs: int,
    device: torch.device,
    max_grad_norm: Optional[float] = None,
    method: str = "euler",
    energy_loss: float = 0.0,
    jacobi_loss: float = 0.0,
    n_traj_val: int = -1,
    loss_method: str = "exact forward",
    loss_iter: int = 5,
    scheme: str = "CN",
    print_every: int = 1
) -> dict:
    """Contains the training and validation loops, generates new data for each batch"""
    history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}

    if isinstance(device, str):
        device = torch.device(device)

    # decays the learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    step_fn = simulator.step_rk4 if method == 'rk4' else simulator.step_euler

    if n_traj_val < 0:
        n_traj_val = n_traj_train // 4
    num_train_batches = max(1, n_traj_train // batch_size)
    num_val_batches = max(1, n_traj_val // batch_size)

    for epoch in range(epochs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        model.train()
        running_losses = {}

        for _ in range(num_train_batches):
            # samples a batch of initial conditions
            z_n = simulator.random_initial_conditions(n_traj=batch_size)
            # creates the targets by simulating with the given step function
            z_n1 = step_fn(z_n, dt)

            z_n = z_n.to(device)
            z_n1 = z_n1.to(device).squeeze(1)
            
            optimizer.zero_grad()
            
            loss_dict = total_loss(model, z_n, z_n1, dt, lambda_jacobi=jacobi_loss, lambda_energy=energy_loss,
                                   method=loss_method, iter=loss_iter, scheme=scheme)
            
            for k, v in loss_dict.items():
                running_losses[f'train_{k}'] = running_losses.get(f'train_{k}', 0.0) + v.item()
                
            total_batch_loss = sum(loss_dict.values())
            total_batch_loss.backward()

            # optionally clip the gradient
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        # add all the losses
        train_sum = 0
        for k in running_losses:
            avg = running_losses[k] / num_train_batches
            if k not in history: history[k] = []
            history[k].append(avg)
            train_sum += avg
        history['train_loss'].append(train_sum)
        
        # evaluation loop
        model.eval()
        running_losses_val = {}
        
        for _ in range(num_val_batches):
            z_n = simulator.random_initial_conditions(n_traj=batch_size)
            z_n1 = step_fn(z_n, dt).squeeze(1)

            v_loss_dict = total_loss(model, z_n, z_n1, dt, lambda_jacobi=jacobi_loss, lambda_energy=energy_loss, 
                                     method=loss_method, iter=loss_iter, scheme=scheme)
            for k, v in v_loss_dict.items():
                running_losses_val[f'val_{k}'] = running_losses_val.get(f'val_{k}', 0.0) + v.item()
        
        val_sum = 0
        for k in running_losses_val:
            avg = running_losses_val[k] / num_val_batches
            if k not in history: history[k] = []
            history[k].append(avg)
            val_sum += avg
        history['val_loss'].append(val_sum)
        
        scheduler.step()

        for param in model.parameters():
            param.requires_grad = True

        if device.type == 'cuda':
            torch.cuda.synchronize()

            #peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            #print("Peak mem: ", peak_mem_mb)
            #torch.cuda.reset_peak_memory_stats(device)
        
        epoch_time = time.time() - start_time
        history['epoch_times'].append(epoch_time)

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_sum:.3e} | Val: {val_sum:.3e} | {epoch_time:.1f}s")

    return history

