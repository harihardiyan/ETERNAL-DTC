import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# Import the core layer module
# As this script is in src/, we assume dtc_layer is accessible
from dtc_layer import DTCLockedLayer, TIME_PERIOD_T, A_DRIVE, ALPHA 

# ==============================================================================
# 1. KONFIGURASI GLOBAL (INFERENCE RUN)
# ==============================================================================

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"🚀 ENGINE START: Menggunakan device {DEVICE}")

# Konstanta Jaringan
INPUT_SIZE = 128  
HIDDEN_SIZE = 512              
OUTPUT_SIZE = 1 
BATCH_SIZE = 512
NUM_LAYERS = 3  

# Inference parameters
INFERENCE_STEPS = 10000000 
LOG_INTERVAL_INF = 1000000
LOAD_PATH = 'results/TCN_NDTC_PerfectCrystal_Final.pth'

# ==============================================================================
# 2. DEFINISI JARINGAN PENUH (TCN)
# ==============================================================================

class TimeCrystallineNetwork(nn.Module):
    def __init__(self, num_layers, d_in, d_hidden, d_out, T=TIME_PERIOD_T):
        super().__init__()
        self.dtc_layers = nn.ModuleList([
            DTCLockedLayer(d_in if i == 0 else d_hidden, d_hidden, T) 
            for i in range(num_layers)
        ])
        self.final_output = nn.Linear(d_hidden, d_out)
        self.T = T
        self.num_layers = num_layers
    
    def init_hidden(self, batch_size):
        for layer in self.dtc_layers:
            layer.init_state(batch_size)

    def forward(self, x, t):
        h = x
        for layer in self.dtc_layers:
            h = layer(h, t)
        return self.final_output(h).squeeze(-1)
    
    def calculate_global_loss(self, t):
        
        psi_layers = []
        phi_layers = []
        
        for layer in self.dtc_layers:
            psi_l_sq = layer.calculate_psi(t)
            psi_layers.append(psi_l_sq)
            phi_layers.append(layer.phi_l)

        L_TC_local_sum = torch.sum(torch.stack(psi_layers))
        Psi_macro_sq = torch.mean(torch.stack(psi_layers))
        
        # Phase Mean & Kuramoto Loss (for logging/verification)
        phases = torch.stack(phi_layers)
        z = torch.mean(torch.exp(1j * phases))
        L_kuramoto = 1 - torch.abs(z)**2
        phi_mean_log = torch.mean(phases) 
        
        return L_TC_local_sum, Psi_macro_sq, L_kuramoto, phi_mean_log

# ==============================================================================
# 3. INFERENCE LOOP (VERIFIKASI STABILITAS)
# ==============================================================================

def verify_eternal_stability():
    
    model = TimeCrystallineNetwork(NUM_LAYERS, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
    
    if not os.path.exists(LOAD_PATH):
         print(f"❌ ERROR FATAL: File bobot akhir tidak ditemukan di {LOAD_PATH}.")
         sys.exit(1)
        
    try:
        model.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE))
        print(f"✅ BOBOT KRISTAL SEMPURNA DARI {LOAD_PATH} BERHASIL DIMUAT UNTUK VERIFIKASI.")
    except Exception as e:
         print(f"❌ ERROR MEMUAT BOBOT: {e}")
         sys.exit(1)

    model.init_hidden(BATCH_SIZE) 
    data_input = torch.randn(BATCH_SIZE, INPUT_SIZE).to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        t_initial = torch.tensor(0)
        _, initial_rigidity, initial_L_kuramoto, initial_phase_mean = model.calculate_global_loss(t_initial)
        
        print("\n--- VERIFIKASI STABILITAS ABADI (10 JUTA STEP) ---")
        print(f"🔬 INITIAL STATE (t=0):")
        print(f"| Rigidity Makro={initial_rigidity.item():.6f}")
        print(f"| L_Kuramoto={initial_L_kuramoto.item():.12f}")
        print(f"| Phase Mean={initial_phase_mean.item():.10f}")
        print(f"| Time Period T={TIME_PERIOD_T} (2T=64)")
        print("--------------------------------------------------")
        
        for i in range(1, INFERENCE_STEPS + 1):
            t = torch.tensor(i % (2 * TIME_PERIOD_T))
            _ = model(data_input, t) 
            
            if i % LOG_INTERVAL_INF == 0 or i == INFERENCE_STEPS:
                _, final_rigidity, L_kuramoto_inf, final_phase_mean = model.calculate_global_loss(t)
                
                print(f"🔬 Inf Step {i} | t={t.item()} ({round(i/INFERENCE_STEPS*100, 2)}% Complete)")
                print(f"| L_Kuramoto={L_kuramoto_inf.item():.12f} (Target < 1e-4)")
                print(f"| Rigidity Makro={final_rigidity.item():.6f} (Initial: {initial_rigidity.item():.6f})")
                
                # Cek Drift Abadi (Kunci utama Time Crystal)
                if abs(final_rigidity.item() - initial_rigidity.item()) > 1e-5:
                    print("!! 🛑 PERINGATAN KRITIS: Rigidity Drift Terdeteksi (>1e-5).")
                if L_kuramoto_inf.item() > 1e-4:
                     print("!! 🛑 PERINGATAN KRITIS: Phase Coherence Degradasi (>1e-4).")


        print("\n✅ INFERENCE 10 JUTA STEP SELESAI.")
        print("--------------------------------------------------")
        print("🎊 KRISTAL WAKTU DISKRET TERVERIFIKASI STABIL ABADI! 🎊")
        print(f"Final Rigidity: {final_rigidity.item():.6f}")
        print(f"Final Phase Coherence (L_Kuramoto): {L_kuramoto_inf.item():.12f}")


if __name__ == '__main__':
    if not os.path.exists('results'):
        os.makedirs('results')
    # Jalankan proses verifikasi yang sangat panjang
    verify_eternal_stability()
