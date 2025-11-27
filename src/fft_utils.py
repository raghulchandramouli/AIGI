import torch
import torch.fft

def apply_fft(x):
    """
    Converts a batch of images (B, C, H, W) to freq domain
    Returns : Amplitude and Phase Tensors
    """

    fft_x = torch.fft.fft2(x, dim=(-2, -1))

    # shift from low to center 
    fft_x = torch.fft.fftshift(fft_x, dim=(-2, -1))

    return fft_x

def apply_ifft(fft_x):

    """
    1. Unshifts from center
    2. inverse FFT
    3. Take the Real part
    """

    fft_x = torch.fft.ifftshift(fft_x, dim=(-2, -1))
    x = torch.fft.ifft2(fft_x, dim=(-2, -1))
    return x.real

def get_mask(batch_size, channel, size, ratio=0.5, device='gpu'):
    """
    bernoulli mask for freq domain
    """

    mask = torch.rand((batch_size, channel, size, size), device=device)
    mask = (mask > ratio).float()
    return mask


def get_spectrum_amplitude(complex_tensor):
    """
    Extracts and normalizes magnitude spectrum from complex tensor.
    """
    amplitude = torch.abs(complex_tensor)
    B = amplitude.shape[0]
    # Flatten spatial dimensions for normalization
    amp_flat = amplitude.reshape(B, -1)
    amp_min = amp_flat.min(dim=1, keepdim=True)[0]
    amp_max = amp_flat.max(dim=1, keepdim=True)[0]
    amp_normalized = (amp_flat - amp_min) / (amp_max - amp_min + 1e-8)
    # Reshape back to original shape
    return amp_normalized.reshape(amplitude.shape)




    