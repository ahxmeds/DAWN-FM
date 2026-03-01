"""
Forward problem operators for image deblurring

This module implements blur operators using FFT for efficient computation.

Reference: https://www.aimsciences.org//article/doi/10.3934/fods.2026005
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class blurFFT(nn.Module):
    """
    Gaussian blur operator implemented in Fourier domain
    
    This operator applies a Gaussian blur using FFT for efficient computation.
    The blur is applied via multiplication in Fourier domain.
    
    Args:
        dim: Image dimension (assumes square images)
        sigma: List of two floats [sigma_x, sigma_y] for Gaussian kernel
        device: Device to run computations ('cuda' or 'cpu')
    """
    def __init__(self, dim=256, sigma=[3, 3], device='cuda'):
        super(blurFFT, self).__init__()
        self.dim = dim
        self.device = device
        self.sigma = sigma
        
        # Pre-compute PSF in Fourier domain
        P, center = self.psfGauss(self.dim)
        S = torch.fft.fft2(torch.roll(P, shifts=center, dims=[0, 1])).unsqueeze(0).unsqueeze(0)
        self.S = S.to(self.device)
    
    def forward(self, I):
        """
        Apply blur operator: B = A * I
        
        Args:
            I: Input image [B, C, H, W]
        
        Returns:
            B: Blurred image [B, C, H, W]
        """
        B = torch.real(torch.fft.ifft2(self.S * torch.fft.fft2(I)))
        return B
    
    def adjoint(self, Ic):
        """
        Apply adjoint of blur operator: I = A^T * Ic
        
        For Gaussian blur, the adjoint is the same as forward operation.
        
        Args:
            Ic: Input image [B, C, H, W]
        
        Returns:
            I: Output image [B, C, H, W]
        """
        I = self.forward(Ic)
        return I
    
    def psfGauss(self, dim):
        """
        Generate Gaussian point spread function (PSF)
        
        Args:
            dim: Dimension of PSF (dim x dim)
        
        Returns:
            PSF: Normalized Gaussian PSF [dim, dim]
            center: Center coordinates for rolling
        """
        s = self.sigma
        m = dim
        n = dim
        
        x = torch.arange(-n//2 + 1, n//2 + 1)
        y = torch.arange(-n//2 + 1, n//2 + 1)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        PSF = torch.exp(-(X**2) / (2*s[0]**2) - (Y**2) / (2*s[1]**2))
        PSF = PSF / torch.sum(PSF)
        
        # Get center ready for output
        center = [1 - m//2, 1 - n//2]
        
        return PSF, center


class blurFFT_generator(nn.Module):
    """
    Gaussian blur with upsampling and downsampling
    
    This operator applies blur by first upsampling by 2x, applying blur in
    Fourier domain, then downsampling back to original size.
    
    Args:
        dim: Image dimension (assumes square images)
        sigma: List of two floats [sigma_x, sigma_y] for Gaussian kernel
        device: Device to run computations ('cuda' or 'cpu')
    """
    def __init__(self, dim=256, sigma=[3, 3], device='cuda'):
        super(blurFFT_generator, self).__init__()
        self.dim = dim
        self.device = device
        self.sigma = sigma
        
        # Create blur operator for upsampled dimension (2x)
        upsampled_dim = dim * 2
        P, center = self.psfGauss(upsampled_dim)
        S = torch.fft.fft2(torch.roll(P, shifts=center, dims=[0, 1])).unsqueeze(0).unsqueeze(0)
        self.S = S.to(self.device)
    
    def forward(self, I):
        """
        Apply blur with upsampling/downsampling
        
        Args:
            I: Input image [B, C, H, W]
        
        Returns:
            B: Blurred image [B, C, H, W]
        """
        # Step 1: Upsample by factor of 2
        I_upsampled = F.interpolate(I, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Step 2: Apply blur in frequency domain
        B_upsampled = torch.real(torch.fft.ifft2(self.S * torch.fft.fft2(I_upsampled)))
        
        # Step 3: Downsample by factor of 2
        B = F.interpolate(B_upsampled, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        return B
    
    def adjoint(self, Ic):
        """
        Apply adjoint operator
        
        Args:
            Ic: Input image [B, C, H, W]
        
        Returns:
            I: Output image [B, C, H, W]
        """
        I = self.forward(Ic)
        return I
    
    def psfGauss(self, dim):
        """Generate Gaussian PSF"""
        s = self.sigma
        m = dim
        n = dim
        
        x = torch.arange(-n//2 + 1, n//2 + 1)
        y = torch.arange(-n//2 + 1, n//2 + 1)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        PSF = torch.exp(-(X**2) / (2*s[0]**2) - (Y**2) / (2*s[1]**2))
        PSF = PSF / torch.sum(PSF)
        
        center = [1 - m//2, 1 - n//2]
        
        return PSF, center


class Tomography(nn.Module):
    """
    Tomography forward operator using Radon transform approximation
    
    This operator computes the filtered backprojection for CT reconstruction.
    It uses a matrix-based approach for the forward projection (sinogram computation)
    and adjoint operation (backprojection).
    
    Args:
        dim: Image dimension (assumes square images)
        num_angles: Number of projection angles (default: 360)
        device: Device to run computations ('cuda' or 'cpu')
    """
    def __init__(self, dim=28, num_angles=360, device='cuda'):
        super(Tomography, self).__init__()
        self.dim = dim
        self.num_angles = num_angles
        self.device = device
        self.pad_size = self.dim // 2
        self.num_detectors = self.dim + 2*self.pad_size + 1
        
        # Create grid for image pixel coordinates (centered at 0)
        X, Y = torch.meshgrid(
            torch.arange(self.dim + 2*self.pad_size, device=self.device) - self.dim, 
            torch.arange(self.dim + 2*self.pad_size, device=self.device) - self.dim,
            indexing='ij'
        )
        self.X = X.float()
        self.Y = Y.float()
        
        # Precompute tomography matrix
        self.A = self.compute_tomography_matrix()
        self.A = self.A.to(self.device)
    
    def compute_tomography_matrix(self):
        """Compute the Radon transform matrix for all angles and detector positions"""
        A_rows = []
        theta = torch.linspace(0, 2*torch.pi, self.num_angles, device=self.device)
        
        for i, angle in enumerate(theta):
            cos_angle = torch.cos(angle)
            sin_angle = torch.sin(angle)
            
            for detector in range(self.num_detectors):
                ray_row = self.compute_ray_row(detector, cos_angle, sin_angle)
                A_rows.append(ray_row.flatten())
        
        A = torch.stack(A_rows, dim=0)
        return A
    
    def compute_ray_row(self, detector_idx, cos_angle, sin_angle):
        """
        Compute the weights for a single ray's interaction with all pixels.
        Uses delta function approximation: 1 if ray intersects pixel, 0 otherwise.
        
        Args:
            detector_idx: Index of the detector position
            cos_angle: Cosine of projection angle
            sin_angle: Sine of projection angle
            
        Returns:
            ray_row: Weight matrix for this ray [dim+2*pad_size, dim+2*pad_size]
        """
        t_vals = torch.linspace(-self.dim, self.dim, steps=self.num_detectors, device=self.device)
        X_rot = self.X * cos_angle + self.Y * sin_angle
        ray_row = (X_rot - t_vals[detector_idx]).abs() < 0.5
        return ray_row.float()
    
    def hamming_filter(self, sinogram):
        """Apply Hamming filter to sinogram in Fourier domain"""
        sinogram_fft = torch.fft.fft(sinogram, dim=3)
        
        # Create Hamming filter
        freqs = torch.fft.fftfreq(self.num_detectors, device=self.device)
        hamming_window = 0.54 + 0.46 * torch.cos(2 * torch.pi * freqs)
        hamming_window = hamming_window.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        filtered_sinogram_fft = sinogram_fft * hamming_window
        filtered_sinogram = torch.real(torch.fft.ifft(filtered_sinogram_fft, dim=3))
        
        return filtered_sinogram
    
    def forward(self, I):
        """
        Forward Radon transform (create sinogram from image)
        
        Args:
            I: Input image [batch_size, 1, dim, dim]
            
        Returns:
            sinogram: Projection data [batch_size, 1, num_angles, num_detectors]
        """
        batch_size = I.shape[0]
        # Pad image to avoid boundary artifacts
        I_pad = F.pad(I, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), 
                      mode='constant', value=0)
        I_flatten = I_pad.view(batch_size, -1)
        
        # Apply forward projection (Radon transform)
        sinogram = torch.matmul(I_flatten, self.A.T)
        sinogram = sinogram.view(batch_size, 1, self.num_angles, self.num_detectors)
        
        return sinogram
    
    def adjoint(self, sinogram, apply_hamming_filter=False):
        """
        Adjoint Radon transform (backprojection from sinogram to image)
        
        Args:
            sinogram: Input sinogram [batch_size, 1, num_angles, num_detectors]
            apply_hamming_filter: Whether to apply Hamming filter before backprojection
            
        Returns:
            image: Backprojected image [batch_size, 1, dim, dim]
        """
        if apply_hamming_filter:
            sinogram = self.hamming_filter(sinogram)
        
        batch_size = sinogram.shape[0]
        sinogram_flatten = sinogram.reshape(batch_size, -1)
        
        # Apply backprojection
        backprojected_image = torch.matmul(sinogram_flatten, self.A)
        backprojected_image = backprojected_image.view(batch_size, 1, 
                                                        self.dim + 2*self.pad_size, 
                                                        self.dim + 2*self.pad_size)
        
        # Remove padding
        return backprojected_image[:, :, self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]


