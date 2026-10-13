"""
ACGAN (Auxiliary Classifier GAN) -- UNSW-NB15 (9 attack classes)
====================================================================================
Mirrors acgan_baseline.py from the NSL-KDD phase. One shared conditional
model trained jointly across all 9 attack classes (Worms, Shellcode,
Backdoor, Analysis, Reconnaissance, DoS, Fuzzers, Exploits, Generic) --
NOT including Normal, same convention as NSL-KDD phase.

Class imbalance here is more extreme than NSL-KDD's (Generic=40,000 vs
Worms=130 -> ~307x ratio, vs NSL-KDD's DoS=42,401 vs U2R=198 -> ~214x).
WeightedRandomSampler still used to give every class roughly equal
batch representation despite this.

Can be run IN PARALLEL with wcgan_baseline_unsw.py -- fully independent
scripts/models/files, just note CPU contention will slow both down
somewhat if no GPU is available.

Author: Vibha J
====================================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "./unsw_preprocessed"
OUTPUT_DIR = "./unsw_acgan_outputs"
LATENT_DIM = 32
LR = 0.0002
BETAS = (0.5, 0.999)
EPOCHS = 500
BATCH_SIZE = 128
SEED = 42

CLASS_FILES = {
    "Worms": "gan_input_Worms.csv",
    "Shellcode": "gan_input_Shellcode.csv",
    "Backdoor": "gan_input_Backdoor.csv",
    "Analysis": "gan_input_Analysis.csv",
    "Reconnaissance": "gan_input_Reconnaissance.csv",
    "DoS": "gan_input_DoS.csv",
    "Fuzzers": "gan_input_Fuzzers.csv",
    "Exploits": "gan_input_Exploits.csv",
    "Generic": "gan_input_Generic.csv",
}
CLASS_NAMES = list(CLASS_FILES.keys())   # index order = label order used by this ACGAN
NUM_CLASSES = len(CLASS_NAMES)

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_combined_data():
    frames, labels = [], []
    for idx, class_name in enumerate(CLASS_NAMES):
        filepath = os.path.join(DATA_DIR, CLASS_FILES[class_name])
        df = pd.read_csv(filepath)
        frames.append(df)
        labels.extend([idx] * len(df))
        print(f"Loaded {class_name}: {len(df)} samples (label={idx})")
    combined_df = pd.concat(frames, axis=0, ignore_index=True)
    labels = np.array(labels)
    feature_cols = combined_df.columns.tolist()
    return combined_df, labels, feature_cols


# ============================================================
# MODELS
# ============================================================
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, output_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 64), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, output_dim), nn.Tanh(),
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat([z, label_input], dim=1)
        return self.net(gen_input)


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2, True), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LeakyReLU(0.2, True), nn.Dropout(0.3),
        )
        self.adv_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.cls_head = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.shared(x)
        validity = self.adv_head(features)
        class_logits = self.cls_head(features)
        return validity, class_logits


# ============================================================
# TRAINING
# ============================================================
def train_acgan():
    combined_df, labels, feature_cols = load_combined_data()
    data = combined_df.values.astype(np.float32)
    n_features = data.shape[1]

    tensor_data = torch.tensor(data)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(tensor_data, tensor_labels)

    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)

    G = Generator(LATENT_DIM, NUM_CLASSES, n_features).to(device)
    D = Discriminator(n_features, NUM_CLASSES).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)
    adv_loss_fn = nn.BCELoss()
    cls_loss_fn = nn.CrossEntropyLoss()

    history = {"g_adv": [], "g_cls": [], "d_adv": [], "d_cls": []}

    print(f"\n{'='*60}\nTraining ACGAN across all {NUM_CLASSES} classes jointly\n{'='*60}")

    for epoch in range(EPOCHS):
        epoch_g_adv, epoch_g_cls = 0.0, 0.0
        epoch_d_adv, epoch_d_cls = 0.0, 0.0
        n_batches = 0

        for real_batch, real_labels in loader:
            real_batch = real_batch.to(device)
            real_labels = real_labels.to(device)
            bs = real_batch.size(0)
            valid = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)

            opt_D.zero_grad()
            d_validity_real, d_cls_real = D(real_batch)
            loss_d_real = adv_loss_fn(d_validity_real, valid) + cls_loss_fn(d_cls_real, real_labels)

            z = torch.randn(bs, LATENT_DIM, device=device)
            gen_labels = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            fake_batch = G(z, gen_labels).detach()
            d_validity_fake, d_cls_fake = D(fake_batch)
            loss_d_fake = adv_loss_fn(d_validity_fake, fake) + cls_loss_fn(d_cls_fake, gen_labels)

            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            opt_D.step()

            opt_G.zero_grad()
            z = torch.randn(bs, LATENT_DIM, device=device)
            gen_labels = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            fake_batch = G(z, gen_labels)
            d_validity, d_cls = D(fake_batch)
            loss_g_adv = adv_loss_fn(d_validity, valid)
            loss_g_cls = cls_loss_fn(d_cls, gen_labels)
            loss_g = loss_g_adv + loss_g_cls
            loss_g.backward()
            opt_G.step()

            epoch_g_adv += loss_g_adv.item()
            epoch_g_cls += loss_g_cls.item()
            epoch_d_adv += (loss_d_real.item() + loss_d_fake.item()) / 2
            epoch_d_cls += cls_loss_fn(d_cls_real, real_labels).item()
            n_batches += 1

        history["g_adv"].append(epoch_g_adv / n_batches)
        history["g_cls"].append(epoch_g_cls / n_batches)
        history["d_adv"].append(epoch_d_adv / n_batches)
        history["d_cls"].append(epoch_d_cls / n_batches)

        if (epoch + 1) % max(1, EPOCHS // 10) == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}]  D_adv: {history['d_adv'][-1]:.4f}  "
                  f"D_cls: {history['d_cls'][-1]:.4f}  G_adv: {history['g_adv'][-1]:.4f}  "
                  f"G_cls: {history['g_cls'][-1]:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history["d_adv"]); axes[0, 0].set_title("Discriminator adversarial loss")
    axes[0, 1].plot(history["d_cls"]); axes[0, 1].set_title("Discriminator classification loss")
    axes[1, 0].plot(history["g_adv"]); axes[1, 0].set_title("Generator adversarial loss")
    axes[1, 1].plot(history["g_cls"]); axes[1, 1].set_title("Generator classification loss")
    for ax in axes.flat:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    plt.suptitle("ACGAN Training Loss -- UNSW-NB15 (9 classes, joint training)")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "loss_curves_acgan.png")
    plt.savefig(plot_path); plt.close()
    print(f"\nSaved loss curves -> {plot_path}")

    return G, feature_cols, combined_df, labels


def generate_and_evaluate(G, feature_cols, combined_df, labels):
    results = []
    G.eval()
    for idx, class_name in enumerate(CLASS_NAMES):
        real_mask = labels == idx
        real_class_df = combined_df.iloc[real_mask].reset_index(drop=True)
        n_samples = len(real_class_df)

        with torch.no_grad():
            z = torch.randn(n_samples, LATENT_DIM, device=device)
            gen_labels = torch.full((n_samples,), idx, dtype=torch.long, device=device)
            synthetic = G(z, gen_labels).cpu().numpy()

        synthetic_df = pd.DataFrame(synthetic, columns=feature_cols)
        synth_path = os.path.join(OUTPUT_DIR, f"synthetic_{class_name}.csv")
        synthetic_df.to_csv(synth_path, index=False)

        real_mean = real_class_df.mean()
        synth_mean = synthetic_df.mean()
        mean_diff = (real_mean - synth_mean).abs().mean()
        print(f"{class_name}: n={n_samples}  mean_abs_diff={mean_diff:.4f}  -> saved {synth_path}")

        results.append({"class_name": class_name, "n_samples": n_samples,
                         "n_features": len(feature_cols), "mean_abs_diff": mean_diff})
    return pd.DataFrame(results)


if __name__ == "__main__":
    G, feature_cols, combined_df, labels = train_acgan()
    summary_df = generate_and_evaluate(G, feature_cols, combined_df, labels)

    print(f"\n{'='*60}\nSUMMARY -- ACGAN Results (UNSW-NB15)\n{'='*60}")
    print(summary_df.to_string(index=False))
    summary_path = os.path.join(OUTPUT_DIR, "acgan_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary -> {summary_path}")

    model_path = os.path.join(OUTPUT_DIR, "generator_acgan.pt")
    torch.save(G.state_dict(), model_path)
    print(f"Saved generator weights -> {model_path}")

    print("\nCompare mean_abs_diff per class against wcgan_summary.csv to decide")
    print("which architecture wins per class, especially for Worms.")