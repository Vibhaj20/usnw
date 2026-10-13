"""
WCGAN (WGAN-GP) -- UNSW-NB15 (9 attack classes)
====================================================================
Mirrors wcgan_baseline.py from the NSL-KDD phase. Built specifically
to fix the discriminator dominance / vanishing gradient problem seen
in the vanilla GAN baseline run on this dataset -- 7 of 9 classes
(Analysis, Reconnaissance, Exploits, Generic, DoS, Fuzzers, Shellcode)
showed the discriminator pulling steadily ahead while G's loss climbed
without bound. Only Worms (130 samples) showed a healthy plateau.

Same fix as before: replace BCE loss with Wasserstein distance +
gradient penalty, so G keeps getting a meaningful gradient even when
D is confidently ahead.

Author: Vibha J
====================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "./unsw_preprocessed"
OUTPUT_DIR = "./unsw_wcgan_outputs"
LATENT_DIM = 32
LR = 0.0001
BETAS = (0.5, 0.9)
N_CRITIC = 5
LAMBDA_GP = 10
SEED = 42

# Same per-class epoch/batch config as the vanilla baseline, for a
# fair, apples-to-apples comparison.
CLASS_CONFIG = {
    "Worms":          {"file": "gan_input_Worms.csv",          "batch_size": 8,   "epochs": 1000},
    "Shellcode":      {"file": "gan_input_Shellcode.csv",      "batch_size": 32,  "epochs": 600},
    "Backdoor":       {"file": "gan_input_Backdoor.csv",       "batch_size": 32,  "epochs": 600},
    "Analysis":       {"file": "gan_input_Analysis.csv",       "batch_size": 32,  "epochs": 600},
    "Reconnaissance": {"file": "gan_input_Reconnaissance.csv", "batch_size": 128, "epochs": 400},
    "DoS":            {"file": "gan_input_DoS.csv",            "batch_size": 128, "epochs": 300},
    "Fuzzers":        {"file": "gan_input_Fuzzers.csv",        "batch_size": 256, "epochs": 250},
    "Exploits":       {"file": "gan_input_Exploits.csv",       "batch_size": 256, "epochs": 200},
    "Generic":        {"file": "gan_input_Generic.csv",        "batch_size": 256, "epochs": 150},
}

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# MODELS
# ============================================================
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, output_dim), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    """No Sigmoid, no BatchNorm -- required for the gradient penalty math."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2, True),
            nn.Linear(128, 64), nn.LeakyReLU(0.2, True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    bs = real_samples.size(0)
    alpha = torch.rand(bs, 1, device=device).expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    grad_outputs = torch.ones(bs, 1, device=device)
    gradients = autograd.grad(
        outputs=critic_interpolates, inputs=interpolates, grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(bs, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ============================================================
# TRAINING LOOP FOR ONE CLASS
# ============================================================
def train_wcgan_for_class(class_name, filepath, batch_size, epochs):
    print(f"\n{'='*60}\nTraining WCGAN (WGAN-GP) for class: {class_name}\n{'='*60}")

    df = pd.read_csv(filepath)
    data = df.values.astype(np.float32)
    n_samples, n_features = data.shape
    print(f"Loaded {n_samples} samples, {n_features} features")

    if n_samples < batch_size:
        print(f"WARNING: batch_size ({batch_size}) > n_samples ({n_samples}). Reducing.")
        batch_size = n_samples

    tensor_data = torch.tensor(data)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    if len(loader) == 0:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    G = Generator(LATENT_DIM, n_features).to(device)
    C = Critic(n_features).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_C = torch.optim.Adam(C.parameters(), lr=LR, betas=BETAS)

    g_losses, c_losses = [], []
    step = 0

    for epoch in range(epochs):
        epoch_g_loss, epoch_c_loss = 0.0, 0.0
        n_g_updates, n_c_updates = 0, 0

        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            bs = real_batch.size(0)

            opt_C.zero_grad()
            z = torch.randn(bs, LATENT_DIM, device=device)
            fake_batch = G(z).detach()
            critic_real = C(real_batch).mean()
            critic_fake = C(fake_batch).mean()
            gp = compute_gradient_penalty(C, real_batch, fake_batch, device)
            loss_c = critic_fake - critic_real + LAMBDA_GP * gp
            loss_c.backward()
            opt_C.step()

            epoch_c_loss += loss_c.item()
            n_c_updates += 1
            step += 1

            if step % N_CRITIC == 0:
                opt_G.zero_grad()
                z = torch.randn(bs, LATENT_DIM, device=device)
                fake_batch = G(z)
                loss_g = -C(fake_batch).mean()
                loss_g.backward()
                opt_G.step()
                epoch_g_loss += loss_g.item()
                n_g_updates += 1

        g_losses.append(epoch_g_loss / max(n_g_updates, 1))
        c_losses.append(epoch_c_loss / max(n_c_updates, 1))

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  Critic_loss: {c_losses[-1]:.4f}  Gen_loss: {g_losses[-1]:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(g_losses, label="Generator loss")
    plt.plot(c_losses, label="Critic loss (Wasserstein estimate)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"WCGAN (WGAN-GP) Training Loss - {class_name}")
    plt.legend(); plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"loss_curve_{class_name}.png")
    plt.savefig(plot_path); plt.close()
    print(f"Saved loss curve -> {plot_path}")

    G.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM, device=device)
        synthetic = G(z).cpu().numpy()
    synthetic_df = pd.DataFrame(synthetic, columns=df.columns)
    synth_path = os.path.join(OUTPUT_DIR, f"synthetic_{class_name}.csv")
    synthetic_df.to_csv(synth_path, index=False)
    print(f"Saved synthetic samples -> {synth_path}")

    real_mean = df.mean()
    synth_mean = synthetic_df.mean()
    mean_diff = (real_mean - synth_mean).abs().mean()
    print(f"Mean absolute difference (real vs synthetic feature means): {mean_diff:.4f}")

    model_path = os.path.join(OUTPUT_DIR, f"generator_{class_name}.pt")
    torch.save(G.state_dict(), model_path)
    print(f"Saved generator weights -> {model_path}")

    return {
        "class_name": class_name, "n_samples": n_samples, "n_features": n_features,
        "final_g_loss": g_losses[-1], "final_c_loss": c_losses[-1], "mean_abs_diff": mean_diff,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results = []
    for class_name, cfg in CLASS_CONFIG.items():
        filepath = os.path.join(DATA_DIR, cfg["file"])
        if not os.path.exists(filepath):
            print(f"SKIPPING {class_name}: file not found at {filepath}")
            continue
        result = train_wcgan_for_class(class_name, filepath, cfg["batch_size"], cfg["epochs"])
        results.append(result)

    print(f"\n{'='*60}\nSUMMARY -- WCGAN (WGAN-GP) Results (UNSW-NB15)\n{'='*60}")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    summary_path = os.path.join(OUTPUT_DIR, "wcgan_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary -> {summary_path}")
    print("\nCompare mean_abs_diff and loss curve SHAPE against baseline_summary.csv.")
    print("Key question: did WGAN-GP fix the discriminator-dominance pattern seen")
    print("in Analysis, Reconnaissance, Exploits, Generic, DoS, Fuzzers, Shellcode?")