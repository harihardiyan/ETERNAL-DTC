import torch
import torch.nn as nn
import numpy as np

# Konfigurasi Fisika
A_DRIVE = 1.0  # Drive Amplitude
ALPHA = 0.5    # Coupling Constant
TIME_PERIOD_T = 32 

class DTCLockedLayer(nn.Module):
    """
    Implements the core layer of the Time Crystalline Network (TCN).
    This layer acts as a driven, non-linear oscillator whose phase (phi_l) 
    is optimized for global coherence.
    """
    def __init__(self, d_in, d_out, T=TIME_PERIOD_T):
        super().__init__()
        self.d_out = d_out
        self.T = T
        
        # 1. Feed-Forward Component
        self.linear = nn.Linear(d_in, d_out)
        
        # 2. Recurrent Oscillator Parameters (W_rec and b_l are optimized for Rigidity)
        self.W_rec = nn.Parameter(torch.randn(d_out, d_out) * 0.1)  
        self.b_l = nn.Parameter(torch.randn(d_out) * 0.1)           
        
        # 3. Phase Order Parameter (phi_l is optimized for Coherence)
        self.phi_l = nn.Parameter(torch.tensor(np.random.rand() * 2 * np.pi)) 
        
        self.h_dtc = None 

    def init_state(self, batch_size):
        """Initializes the internal oscillator state (h_dtc) with small noise."""
        device = self.W_rec.device
        dummy_input = torch.zeros(batch_size, self.d_out, device=device) 
        self.h_dtc = torch.randn_like(dummy_input) * 0.1 

    def forward(self, h_in, t):
        if self.h_dtc is None or self.h_dtc.size(0) != h_in.size(0):
             self.init_state(h_in.size(0))

        # Drive: A * cos(pi*t/T + phi_l)
        drive = A_DRIVE * torch.cos(torch.pi * t / self.T + self.phi_l) 
        
        # Internal Oscillation Step (Core DTC Dynamics)
        h_dtc_next = torch.tanh(self.h_dtc @ self.W_rec + drive * self.b_l.unsqueeze(0))
        
        # Coupling to Feed-Forward Input
        h_out_next = ALPHA * h_dtc_next + (1 - ALPHA) * self.linear(h_in)
        
        self.h_dtc = h_dtc_next # Update state
        return h_out_next

    def calculate_psi(self, t):
        """Calculates Local Rigidity (Psi) for the layer."""
        if self.h_dtc is None:
            return torch.tensor(0.0, device=self.W_rec.device) 
        
        # Rigidity: Mean squared magnitude of the internal state
        psi_mag = torch.mean(torch.abs(self.h_dtc), dim=-1)
        return torch.mean(psi_mag**2)
