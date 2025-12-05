
# tcn_dtc_full_colab.py
import math, time, csv, os, json
from typing import Tuple, List, Optional

import torch
import torch.nn as nn

# =========================
# Konfigurasi default aman
# =========================
class TCNConfig:
    def __init__(
        self,
        L: int = 3,
        Dh: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
        A: float = 1.0,
        T: int = 32,
        alpha: float = 0.5,
        lambda_rigidity: float = 0.01,
        W_coh: float = 100.0,
        max_phase_abs: float = 10.0,
        weight_norm_limit: float = 5.0,
        grad_clip_norm: float = 1.0,
        steps: int = 1000,
        batch_size: int = 32,
        log_every: int = 50,
        eval_fft_every: int = 200,
        fft_window: int = 512,
        run_dir: str = "runs",
        seed: int = 42,
        do_ablation: bool = True,
        ablate_at_step: int = 600,
        do_robustness: bool = True,
        robustness_at_step: int = 800,
        robustness_noise_sigma: float = 0.02,
        perturbation_scale: float = 0.01,
        enable_plot: bool = True,
    ):
        self.L = L
        self.Dh = Dh
        self.device = device
        self.dtype = dtype
        self.A = A
        self.T = T
        self.alpha = alpha
        self.lambda_rigidity = lambda_rigidity
        self.W_coh = W_coh
        self.max_phase_abs = max_phase_abs
        self.weight_norm_limit = weight_norm_limit
        self.grad_clip_norm = grad_clip_norm
        self.steps = steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.eval_fft_every = eval_fft_every
        self.fft_window = fft_window
        self.run_dir = run_dir
        self.seed = seed
        self.do_ablation = do_ablation
        self.ablate_at_step = ablate_at_step
        self.do_robustness = do_robustness
        self.robustness_at_step = robustness_at_step
        self.robustness_noise_sigma = robustness_noise_sigma
        self.perturbation_scale = perturbation_scale
        self.enable_plot = enable_plot

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_run_path(base_dir: str) -> str:
    tstamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, tstamp)
    os.makedirs(path, exist_ok=True)
    return path

def assert_finite_tensor(t: torch.Tensor, name: str):
    if not torch.isfinite(t).all():
        raise RuntimeError(f"Terdapat nilai non-finite pada tensor {name}")

# =========================
# Layer dan Model
# =========================
class DTCLockedLayer(nn.Module):
    def __init__(self, Dh: int, alpha: float, A: float, T: int, dtype: torch.dtype):
        super().__init__()
        self.Dh = Dh
        self.alpha = alpha
        self.A = A
        self.T = T

        self.W_rec = nn.Parameter(torch.empty(Dh, Dh, dtype=dtype))
        self.W_feed = nn.Parameter(torch.empty(Dh, Dh, dtype=dtype))

        self.phi = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(Dh, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.W_rec, gain=1.0)
        nn.init.orthogonal_(self.W_feed, gain=1.0)
        nn.init.uniform_(self.b, a=-0.01, b=0.01)
        with torch.no_grad():
            self.phi.data.zero_()

    @torch.no_grad()
    def constrain_weights(self, max_norm: float):
        for W in [self.W_rec, self.W_feed]:
            n = torch.norm(W)
            if torch.isfinite(n) and n > max_norm:
                W.mul_(max_norm / (n + 1e-12))

    def forward(self, h_t: torch.Tensor, x_t: torch.Tensor, t_int: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B = h_t.size(0)
        dtype = h_t.dtype
        device = h_t.device

        phi_clamped = torch.clamp(self.phi, -10.0, 10.0)
        phase = (math.pi * float(t_int) / float(self.T)) + phi_clamped
        phase_val = torch.cos(phase).to(dtype=dtype).expand(B, 1)

        drive = self.A * phase_val * self.b.to(device=device, dtype=dtype)

        h_next_lin = torch.matmul(h_t, self.W_rec) + drive
        h_next = torch.tanh(h_next_lin)

        x_next = self.alpha * h_next + (1.0 - self.alpha) * torch.matmul(x_t, self.W_feed)
        return h_next, x_next

class TCNModel(nn.Module):
    def __init__(self, cfg: TCNConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([
            DTCLockedLayer(cfg.Dh, cfg.alpha, cfg.A, cfg.T, cfg.dtype)
            for _ in range(cfg.L)
        ])

    @torch.no_grad()
    def constrain_all(self):
        for layer in self.layers:
            layer.constrain_weights(self.cfg.weight_norm_limit)
            layer.phi.data.clamp_(-self.cfg.max_phase_abs, self.cfg.max_phase_abs)

    def forward_step(self, h_t_list: torch.Tensor, x_t_list: torch.Tensor, t_int: int) -> Tuple[torch.Tensor, torch.Tensor]:
        L, B, Dh = h_t_list.shape
        h_next_list = torch.empty_like(h_t_list)
        x_next_list = torch.empty_like(x_t_list)
        for l, layer in enumerate(self.layers):
            h_next, x_next = layer(h_t_list[l], x_t_list[l], t_int)
            h_next_list[l] = h_next
            x_next_list[l] = x_next
        return h_next_list, x_next_list

    def phases_vector(self) -> torch.Tensor:
        return torch.cat([layer.phi.view(1) for layer in self.layers], dim=0)

# =========================
# Order parameter dan loss
# =========================
def psi_makro(h_next_list: torch.Tensor) -> torch.Tensor:
    # |h|^2 rata-rata di seluruh (L,B,Dh)
    return h_next_list.pow(2).mean()

def kuramoto_loss(phases: torch.Tensor) -> torch.Tensor:
    Zx = torch.cos(phases).mean()
    Zy = torch.sin(phases).mean()
    return 1.0 - (Zx * Zx + Zy * Zy)

def total_loss(psi: torch.Tensor, L_kura: torch.Tensor, cfg: TCNConfig) -> torch.Tensor:
    return -cfg.lambda_rigidity * psi + cfg.W_coh * L_kura

# =========================
# FFT dan Autokorelasi
# =========================
def compute_fft(signal: torch.Tensor, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    N = signal.numel()
    window = torch.hann_window(N, device=device, dtype=signal.dtype)
    sig_win = (signal - signal.mean()) * window
    fft = torch.fft.rfft(sig_win)
    mag = torch.abs(fft)
    freqs = torch.fft.rfftfreq(N)
    return freqs, mag

def dominant_period_from_fft(freqs: torch.Tensor, mag: torch.Tensor) -> Optional[float]:
    mask = freqs > 0
    if mask.sum() == 0:
        return None
    freqs_nz = freqs[mask]
    mag_nz = mag[mask]
    idx = torch.argmax(mag_nz)
    f_dom = float(freqs_nz[idx])
    if f_dom <= 0:
        return None
    return 1.0 / f_dom

# =========================
# Logger CSV
# =========================
class StepLogger:
    def __init__(self, dirpath: str, L: int):
        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)
        self.file = open(os.path.join(dirpath, "log.csv"), "w", newline="")
        self.writer = csv.writer(self.file)
        header = ["step", "psi", "kuramoto", "loss"] + [f"phi_{l}" for l in range(L)]
        self.writer.writerow(header)
        self.file.flush()

    def write(self, step: int, psi: float, kura: float, loss: float, phases: List[float]):
        row = [step, psi, kura, loss] + phases
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()

# =========================
# Training/Ablation/Robustness
# =========================
def train_run(cfg: TCNConfig):
    set_seed(cfg.seed)

    run_path = make_run_path(cfg.run_dir)
    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump({
            "L": cfg.L, "Dh": cfg.Dh, "device": cfg.device, "dtype": str(cfg.dtype),
            "A": cfg.A, "T": cfg.T, "alpha": cfg.alpha,
            "lambda_rigidity": cfg.lambda_rigidity, "W_coh": cfg.W_coh,
            "steps": cfg.steps, "batch_size": cfg.batch_size
        }, f, indent=2)
    print(f"Run directory: {run_path}")

    model = TCNModel(cfg).to(device=cfg.device, dtype=cfg.dtype)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    L = cfg.L
    B = cfg.batch_size
    Dh = cfg.Dh

    h_t = torch.zeros(L, B, Dh, dtype=cfg.dtype, device=cfg.device)
    x_t = torch.zeros(L, B, Dh, dtype=cfg.dtype, device=cfg.device)

    logger = StepLogger(run_path, L)

    # Trace untuk FFT
    trace_len = cfg.fft_window
    unit_trace = torch.zeros(trace_len, dtype=cfg.dtype, device=cfg.device)
    trace_idx = 0
    layer_to_trace = 0
    unit_to_trace = 0

    for t in range(cfg.steps):
        if t % 10 == 0:
            with torch.no_grad():
                model.constrain_all()

        # Perturbasi robustness pada langkah tertentu
        drive_noise = None
        if cfg.do_robustness and t == cfg.robustness_at_step:
            drive_noise = torch.randn(B, Dh, device=cfg.device, dtype=cfg.dtype) * cfg.robustness_noise_sigma
            with torch.no_grad():
                for layer in model.layers:
                    layer.W_rec.add_(cfg.perturbation_scale * torch.randn_like(layer.W_rec))

        h_next, x_next = model.forward_step(h_t, x_t, t)

        if drive_noise is not None:
            h_next = torch.tanh(h_next + drive_noise)

        psi = psi_makro(h_next)
        phases = model.phases_vector()
        L_kura = kuramoto_loss(phases)
        loss = total_loss(psi, L_kura, cfg)

        assert_finite_tensor(psi, "Psi_Makro")
        assert_finite_tensor(L_kura, "Kuramoto_Loss")
        assert_finite_tensor(loss, "Total_Loss")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        opt.step()

        h_t, x_t = h_next.detach(), x_next.detach()

        # Rekam trace satu unit
        if trace_idx < trace_len:
            unit_trace[trace_idx] = h_t[layer_to_trace, 0, unit_to_trace]
            trace_idx += 1

        # Logging
        if (t + 1) % cfg.log_every == 0 or t == 0:
            vals = [float(psi.item()), float(L_kura.item()), float(loss.item())]
            phivals = [float(p.item()) for p in phases]
            logger.write(t + 1, vals[0], vals[1], vals[2], phivals)
            print(f"[t={t+1}] psi={vals[0]:.6f} | kura={vals[1]:.6f} | loss={vals[2]:.6f}")

        # FFT evaluasi
        if (t + 1) % cfg.eval_fft_every == 0 and trace_idx == trace_len:
            freqs, mag = compute_fft(unit_trace, cfg.device)
            period = dominant_period_from_fft(freqs, mag)
            torch.save({"freqs": freqs.cpu(), "mag": mag.cpu(), "period": period},
                       os.path.join(run_path, f"fft_t{t+1}.pt"))
            if period is not None:
                print(f"FFT@{t+1}: dominant period ≈ {period:.2f} steps (target ~ {2*cfg.T})")
            unit_trace.zero_(); trace_idx = 0

        # Ablation: set lambda = 0
        if cfg.do_ablation and (t + 1) == cfg.ablate_at_step:
            print(f"[Ablation] lambda_rigidity -> 0 at step {t+1}")
            cfg.lambda_rigidity = 0.0

        if not torch.isfinite(h_t).all() or not torch.isfinite(x_t).all():
            logger.close()
            raise RuntimeError("State mengandung NaN/Inf — berhenti untuk stabilitas.")

    logger.close()
    torch.save({"state_dict": model.state_dict(),
                "cfg": {"L": cfg.L, "Dh": cfg.Dh, "A": cfg.A, "T": cfg.T, "alpha": cfg.alpha}},
               os.path.join(run_path, "model_final.pt"))
    print("Training selesai.")

    # Autokorelasi opsional (bisa dipakai untuk plotting)
    if trace_idx == trace_len:
        ac_fft = torch.fft.fft((unit_trace - unit_trace.mean()))
        S = ac_fft * torch.conj(ac_fft)
        ac = torch.real(torch.fft.ifft(S))
        ac = ac / (ac.max() + 1e-12)
        torch.save({"autocorr": ac.cpu()}, os.path.join(run_path, "autocorr.pt"))
        print("Autokorelasi disimpan.")

# =========================
# Plot utilities (opsional)
# =========================
def plot_quick(run_dir: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib tidak tersedia—lewati plotting.")
        return
    log_path = os.path.join(run_dir, "log.csv")
    if not os.path.exists(log_path):
        print("log.csv tidak ditemukan.")
        return
    # Baca log
    steps, psi, kura, loss = [], [], [], []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            psi.append(float(row["psi"]))
            kura.append(float(row["kuramoto"]))
            loss.append(float(row["loss"]))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(steps, psi, label="Psi"); plt.title("Rigiditas (Psi)"); plt.xlabel("Step"); plt.ylabel("Psi"); plt.grid(True)
    plt.subplot(1,2,2); plt.plot(steps, kura, label="Kuramoto"); plt.title("Koherensi (Kuramoto loss)"); plt.xlabel("Step"); plt.ylabel("Loss"); plt.grid(True)
    plt.tight_layout(); plt.show()

# =========================
# Main
# =========================
if __name__ == "__main__":
    cfg = TCNConfig(steps=1000, batch_size=32, device="cpu")  # default aman untuk uji coba
    run_path = make_run_path(cfg.run_dir)
    # Pastikan logger dan artefak ke folder run ini
    # override run_dir sementara
    cfg.run_dir = run_path
    train_run(cfg)
    if cfg.enable_plot:
        plot_quick(run_path)
