"""
Vanilla GAN Baseline -- UNSW-NB15 (9 attack classes)
====================================================================
Mirrors vanilla_gan_baseline.py from the NSL-KDD phase. De-risks
training stability before WCGAN/ACGAN, now across 9 classes instead
of 4, with 69 features instead of 101.

Worms (130 samples) is the new most-extreme minority class -- more
conservative batch/epoch treatment than U2R got (198 samples) in the
NSL-KDD phase, since there's even less real data to work with.

Generic is already at the undersampling target (40,000) and won't
need synthetic samples at merge time, but is trained here anyway for
a complete, consistent comparative analysis across all classes.

Author: Vibha J
====================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "./unsw_preprocessed"
OUTPUT_DIR = "./unsw_gan_baseline_outputs"
LATENT_DIM = 32
LR = 0.0002
SEED = 42

# Per-class config: smaller class -> smaller batch, more epochs.
# Worms gets the most conservative treatment of all (only 130 samples).
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
# MODELS (identical architecture to NSL-KDD phase -- only input/output
# dimensions change automatically based on the data)
# ============================================================
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, output_dim), nn.Tanh(),  # matches [-1,1] scaling
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2, True), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LeakyReLU(0.2, True), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# TRAINING LOOP FOR ONE CLASS
# ============================================================
def train_gan_for_class(class_name, filepath, batch_size, epochs):
    print(f"\n{'='*60}\nTraining vanilla GAN for class: {class_name}\n{'='*60}")

    df = pd.read_csv(filepath)
    data = df.values.astype(np.float32)
    n_samples, n_features = data.shape
    print(f"Loaded {n_samples} samples, {n_features} features")

    if n_samples < batch_size:
        print(f"WARNING: batch_size ({batch_size}) > n_samples ({n_samples}). "
              f"Reducing batch_size to {n_samples}.")
        batch_size = n_samples

    tensor_data = torch.tensor(data)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    G = Generator(LATENT_DIM, n_features).to(device)
    D = Discriminator(n_features).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    g_losses, d_losses = [], []

    for epoch in range(epochs):
        epoch_g_loss, epoch_d_loss = 0.0, 0.0
        n_batches = 0

        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            bs = real_batch.size(0)
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            opt_D.zero_grad()
            d_real = D(real_batch)
            loss_d_real = criterion(d_real, real_labels)
            z = torch.randn(bs, LATENT_DIM, device=device)
            fake_batch = G(z).detach()
            d_fake = D(fake_batch)
            loss_d_fake = criterion(d_fake, fake_labels)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_D.step()

            opt_G.zero_grad()
            z = torch.randn(bs, LATENT_DIM, device=device)
            fake_batch = G(z)
            d_fake_for_g = D(fake_batch)
            loss_g = criterion(d_fake_for_g, real_labels)
            loss_g.backward()
            opt_G.step()

            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            n_batches += 1

        g_losses.append(epoch_g_loss / n_batches)
        d_losses.append(epoch_d_loss / n_batches)

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {d_losses[-1]:.4f}  G_loss: {g_losses[-1]:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(g_losses, label="Generator loss")
    plt.plot(d_losses, label="Discriminator loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Vanilla GAN Training Loss - {class_name}")
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
        "final_g_loss": g_losses[-1], "final_d_loss": d_losses[-1], "mean_abs_diff": mean_diff,
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
        result = train_gan_for_class(class_name, filepath, cfg["batch_size"], cfg["epochs"])
        results.append(result)

    print(f"\n{'='*60}\nSUMMARY -- Vanilla GAN Baseline Results (UNSW-NB15)\n{'='*60}")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    summary_path = os.path.join(OUTPUT_DIR, "baseline_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary -> {summary_path}")
    print("\nNEXT STEP: Inspect loss_curve_*.png for each class, especially Worms")
    print("(most extreme minority class -- 130 real samples). Once confirmed stable,")
    print("move to WCGAN for all 9 classes.")