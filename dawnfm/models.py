"""
Network architectures for DAW-FM and DAWN-FM

This module contains:
- UNetFMG_DE: U-Net with data embedding only (for DAW-FM)
- UNetFMG_DE_NE: U-Net with data and noise embedding (for DAWN-FM)
- ODE solvers for inference

Reference: https://www.aimsciences.org//article/doi/10.3934/fods.2026005
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList


# ============================================================================
# ODE Solvers for Inference
# ============================================================================

def odeSol_data(x0, ATb, model, nsteps=100):
    """
    ODE solver for data embedding only (DAW-FM)
    
    Args:
        x0: Initial condition (noise)
        ATb: Adjoint of blurred data
        model: Neural network model
        nsteps: Number of integration steps
    
    Returns:
        traj: Trajectory of solutions [nsteps+1, B, C, H, W]
    """
    traj = torch.zeros(nsteps+1, x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[3], 
                      device=x0.device)
    traj[0, :, :, :, :] = x0
    t = torch.zeros(x0.shape[0], device=x0.device)
    
    with torch.no_grad():
        h = 1 / nsteps
        for i in range(nsteps):
            xt = traj[i, :, :, :, :]
            k1 = model(xt, t, ATb)
            k2 = model(xt + h*k1/2, t + h/2, ATb)
            k3 = model(xt + h*k2/2, t + h/2, ATb)
            k4 = model(xt + h*k3, t + h, ATb)
            traj[i+1, :, :, :, :] = traj[i, :, :, :, :] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t = t + h
    
    return traj


def odeSol_data_noise(x0, ATb, sigma, model, nsteps=100):
    """
    ODE solver for data and noise embedding (DAWN-FM)
    
    Args:
        x0: Initial condition (noise)
        ATb: Adjoint of blurred data
        sigma: Noise standard deviation
        model: Neural network model
        nsteps: Number of integration steps
    
    Returns:
        traj: Trajectory of solutions [nsteps+1, B, C, H, W]
    """
    traj = torch.zeros(nsteps+1, x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[3],
                      device=x0.device)
    traj[0, :, :, :, :] = x0
    t = torch.zeros(x0.shape[0], device=x0.device)
    
    with torch.no_grad():
        h = 1 / nsteps
        for i in range(nsteps):
            xt = traj[i, :, :, :, :]
            k1 = model(xt, t, ATb, sigma)
            k2 = model(xt + h*k1/2, t + h/2, ATb, sigma)
            k3 = model(xt + h*k2/2, t + h/2, ATb, sigma)
            k4 = model(xt + h*k3, t + h, ATb, sigma)
            traj[i+1, :, :, :, :] = traj[i, :, :, :, :] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t = t + h
    
    return traj


# ============================================================================
# Helper Functions and Modules
# ============================================================================

def Id(nc):
    """Create identity convolution"""
    conv = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, 
                    padding=1, groups=nc, bias=False)
    
    # Initialize filters to identity
    with torch.no_grad():
        for i in range(nc):
            conv.weight[i, 0, :, :] = torch.zeros((3, 3))
            conv.weight[i, 0, 1, 1] = 1.0
    return conv


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class RMSNorm(Module):
    """
    RMS normalization across channel dimension
    
    Input: N x C x H x W
    Output: N x C x H x W
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class UnetBlock2D(nn.Module):
    """Basic U-Net block with layer normalization and convolutions"""
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super(UnetBlock2D, self).__init__()
        self.layerNorm1 = nn.LayerNorm(shape)
        self.layerNorm2 = nn.LayerNorm([out_c, shape[1], shape[2]])
        
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        
        self.activation = nn.SiLU()
    
    def forward(self, x):
        out = self.layerNorm1(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.layerNorm2(out)
        out = self.conv3(out)
        out = self.activation(out)
        return out


class resnetBlock(nn.Module):
    """ResNet-style block with skip connections"""
    def __init__(self, dims, in_c, out_c, levels=5):
        super(resnetBlock, self).__init__()
        
        self.Open = UnetBlock2D((in_c, dims[0], dims[1]), in_c, out_c)
        self.Blocks = nn.ModuleList()
        for i in range(levels):
            bi = UnetBlock2D((out_c, dims[0], dims[1]), out_c, out_c)
            self.Blocks.append(bi)
    
    def forward(self, x):
        x = self.Open(x)
        for i in range(len(self.Blocks)):
            x = x + self.Blocks[i](x)
        return x


# ============================================================================
# UNetFMG_DE: Data Embedding Only (DAW-FM)
# ============================================================================

class UNetFMG_DE(nn.Module):
    """
    U-Net with Full Multi-Grid and Data Embedding
    
    This network uses only data embedding (adjoint of blurred data).
    Used for DAW-FM (Data-Aware Flow Matching).
    
    Args:
        arch: List of channel numbers at each level (e.g., [1, 16, 32])
        dims: Tensor of spatial dimensions [H, W]
        time_emb_dim: Dimension of time embedding
    """
    def __init__(self, arch=[3, 16, 32, 64, 128], dims=torch.tensor([64, 64]), 
                 time_emb_dim=256):
        super(UNetFMG_DE, self).__init__()
        
        self.time_embed = nn.Linear(1, time_emb_dim)
        
        # Down blocks
        self.DBlocks = nn.ModuleList()
        self.DTE = nn.ModuleList()  # Down time embeddings
        self.DDE = nn.ModuleList()  # Down data embeddings
        
        for i in range(len(arch) - 1):
            te = self._make_te(time_emb_dim, arch[i])
            de = self._make_de(arch[0], arch[i])
            blk = resnetBlock((dims[0], dims[1]), arch[i], arch[i+1])
            
            self.DBlocks.append(blk)
            self.DTE.append(te)
            self.DDE.append(de)
            dims = dims // 2
        
        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, arch[-1])
        self.blk_mid = resnetBlock((dims[0], dims[1]), arch[-1], arch[-1])
        
        # Up blocks
        self.UBlocks = nn.ModuleList()
        self.UTE = nn.ModuleList()  # Up time embeddings
        self.UDE = nn.ModuleList()  # Up data embeddings
        self.Smth = nn.ModuleList()  # Smoothing operators
        
        for i in np.flip(range(len(arch) - 1)):
            dims = dims * 2
            teu = self._make_te(time_emb_dim, arch[i+1])
            de = self._make_de(arch[0], arch[i+1])
            blku = resnetBlock((dims[0], dims[1]), arch[i+1], arch[i])
            
            self.Smth.append(Id(arch[i]))
            self.UBlocks.append(blku)
            self.UTE.append(teu)
            self.UDE.append(de)
        
        self.h = nn.Parameter(torch.ones(len(arch)))
    
    def Coarsen(self, x):
        """Downsample by factor of 2"""
        return F.interpolate(x, scale_factor=0.5)
    
    def Refine(self, x):
        """Upsample by factor of 2"""
        return F.interpolate(x, scale_factor=2, mode='bilinear')
    
    def forward(self, x, t, ATb):
        """
        Forward pass
        
        Args:
            x: Input [B, C, H, W]
            t: Time [B]
            ATb: Adjoint of blurred data [B, C, H, W]
        
        Returns:
            Output [B, C, H, W]
        """
        t = self.time_embed(t.unsqueeze(1))
        n = len(x)
        
        # Downsampling path
        X = [x]
        for i in range(len(self.DBlocks)):
            scale = x.shape[-1] / ATb.shape[-1]
            ATbI = F.interpolate(ATb, scale_factor=scale)
            de = self.DDE[i](ATbI)
            te = self.DTE[i](t).reshape(n, -1, 1, 1)
            
            x = self.DBlocks[i](x + te + de)
            x = self.Coarsen(x)
            X.append(x)
        
        # Bottleneck
        te_mid = self.te_mid(t).reshape(n, -1, 1, 1)
        dx = self.blk_mid(x + te_mid)
        x = X[-1] + dx
        
        # Upsampling path
        cnt = -1
        for i in range(len(self.DBlocks)):
            cnt = cnt - 1
            x = self.Refine(x)
            te = self.UTE[i](t).reshape(n, -1, 1, 1)
            scale = x.shape[-1] / ATb.shape[-1]
            ATbI = F.interpolate(ATb, scale_factor=scale)
            de = self.UDE[i](ATbI)
            
            x = self.Smth[i](X[cnt]) + self.h[i] * self.UBlocks[i](x + te + de)
        
        return x
    
    def _make_te(self, dim_in, dim_out):
        """Create time embedding MLP"""
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
    
    def _make_de(self, dim_in, dim_out):
        """Create data embedding CNN"""
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)
        )


# ============================================================================
# UNetFMG_DE_NE: Data and Noise Embedding (DAWN-FM)
# ============================================================================

class UNetFMG_DE_NE(nn.Module):
    """
    U-Net with Full Multi-Grid, Data Embedding, and Noise Embedding
    
    This network uses both data embedding and noise embedding.
    Used for DAWN-FM (Data-Aware Noise-Embedded Flow Matching).
    
    Args:
        arch: List of channel numbers at each level (e.g., [1, 16, 32])
        dims: Tensor of spatial dimensions [H, W]
        time_emb_dim: Dimension of time embedding
        noise_emb_dim: Dimension of noise embedding
    """
    def __init__(self, arch=[3, 16, 32, 64, 128], dims=torch.tensor([64, 64]),
                 time_emb_dim=256, noise_emb_dim=256):
        super(UNetFMG_DE_NE, self).__init__()
        
        self.time_embed = nn.Linear(1, time_emb_dim)
        self.noise_embed = nn.Linear(1, noise_emb_dim)
        
        # Down blocks
        self.DBlocks = nn.ModuleList()
        self.DTE = nn.ModuleList()  # Down time embeddings
        self.DDE = nn.ModuleList()  # Down data embeddings
        self.DNE = nn.ModuleList()  # Down noise embeddings
        
        for i in range(len(arch) - 1):
            te = self._make_te(time_emb_dim, arch[i])
            ne = self._make_te(noise_emb_dim, arch[i])
            de = self._make_de(arch[0], arch[i])
            blk = resnetBlock((dims[0], dims[1]), arch[i], arch[i+1])
            
            self.DBlocks.append(blk)
            self.DTE.append(te)
            self.DDE.append(de)
            self.DNE.append(ne)
            dims = dims // 2
        
        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, arch[-1])
        self.ne_mid = self._make_te(noise_emb_dim, arch[-1])
        self.blk_mid = resnetBlock((dims[0], dims[1]), arch[-1], arch[-1])
        
        # Up blocks
        self.UBlocks = nn.ModuleList()
        self.UTE = nn.ModuleList()  # Up time embeddings
        self.UDE = nn.ModuleList()  # Up data embeddings
        self.UNE = nn.ModuleList()  # Up noise embeddings
        self.Smth = nn.ModuleList()  # Smoothing operators
        
        for i in np.flip(range(len(arch) - 1)):
            dims = dims * 2
            teu = self._make_te(time_emb_dim, arch[i+1])
            neu = self._make_te(noise_emb_dim, arch[i+1])
            de = self._make_de(arch[0], arch[i+1])
            blku = resnetBlock((dims[0], dims[1]), arch[i+1], arch[i])
            
            self.Smth.append(Id(arch[i]))
            self.UBlocks.append(blku)
            self.UTE.append(teu)
            self.UNE.append(neu)
            self.UDE.append(de)
        
        self.h = nn.Parameter(torch.ones(len(arch)))
    
    def Coarsen(self, x):
        """Downsample by factor of 2"""
        return F.interpolate(x, scale_factor=0.5)
    
    def Refine(self, x):
        """Upsample by factor of 2"""
        return F.interpolate(x, scale_factor=2, mode='bilinear')
    
    def forward(self, x, t, ATb, noise):
        """
        Forward pass
        
        Args:
            x: Input [B, C, H, W]
            t: Time [B]
            ATb: Adjoint of blurred data [B, C, H, W]
            noise: Noise level (standard deviation) [B]
        
        Returns:
            Output [B, C, H, W]
        """
        t = self.time_embed(t.unsqueeze(1))
        noise = self.noise_embed(noise.unsqueeze(1))
        n = len(x)
        
        # Downsampling path
        X = [x]
        for i in range(len(self.DBlocks)):
            scale = x.shape[-1] / ATb.shape[-1]
            ATbI = F.interpolate(ATb, scale_factor=scale)
            de = self.DDE[i](ATbI)
            te = self.DTE[i](t).reshape(n, -1, 1, 1)
            ne = self.DNE[i](noise).reshape(n, -1, 1, 1)
            
            x = self.DBlocks[i](x + te + de + ne)
            x = self.Coarsen(x)
            X.append(x)
        
        # Bottleneck
        te_mid = self.te_mid(t).reshape(n, -1, 1, 1)
        ne_mid = self.ne_mid(noise).reshape(n, -1, 1, 1)
        dx = self.blk_mid(x + te_mid + ne_mid)
        x = X[-1] + dx
        
        # Upsampling path
        cnt = -1
        for i in range(len(self.DBlocks)):
            cnt = cnt - 1
            x = self.Refine(x)
            scale = x.shape[-1] / ATb.shape[-1]
            ATbI = F.interpolate(ATb, scale_factor=scale)
            te = self.UTE[i](t).reshape(n, -1, 1, 1)
            ne = self.UNE[i](noise).reshape(n, -1, 1, 1)
            de = self.UDE[i](ATbI)
            
            x = self.Smth[i](X[cnt]) + self.h[i] * self.UBlocks[i](x + te + de + ne)
        
        return x
    
    def _make_te(self, dim_in, dim_out):
        """Create time/noise embedding MLP"""
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
    
    def _make_de(self, dim_in, dim_out):
        """Create data embedding CNN"""
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)
        )
