import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv, os, sys


# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.path.join(BASE_DIR, 'base_racing_line.csv')
OUTPUT_PATH  = os.path.join(BASE_DIR, 'master_brain.pth')
EPOCHS       = 350            # Quick train. Use 3000 for full overnight train.
LR           = 0.001          # Higher LR converges faster in fewer epochs
VAL_SPLIT    = 0.15
BATCH_SIZE   = 512
MAX_PATIENCE = 100            # Early stop if val loss flat for this many epochs
LOSS_WEIGHTS = torch.tensor([1.0, 1.5, 3.0])  # [accel, brake, steer]

print(f"Loading {CSV_PATH}...")

if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found.")
    print("Run SetLine.py first to record a lap, or check the filename.")
    sys.exit(1)

X_list, Y_list = [], []
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            track  = [float(row[f'track_{i}']) for i in range(19)]
            speedX = float(row['speedX'])
            tp     = float(row['trackPos'])
            ang    = float(row['angle'])
            accel  = float(row['accel'])
            brake  = float(row['brake'])
            steer  = float(row['steer'])

            # Normalise inputs to roughly [-1, 1]
            inp = [t / 200.0 for t in track] + [speedX / 200.0, tp, ang]

            # Clamp outputs to tanh range
            out = [
                float(np.clip(accel, -1.0, 1.0)),
                float(np.clip(brake, -1.0, 1.0)),
                float(np.clip(steer, -1.0, 1.0)),
            ]

            X_list.append(inp)
            Y_list.append(out)
        except:
            continue

if len(X_list) < 100:
    print(f"ERROR: Only {len(X_list)} valid rows found in {CSV_PATH}.")
    print("The file may be empty or have wrong column names.")
    sys.exit(1)

print(f"Loaded {len(X_list)} frames.")

X = np.array(X_list, dtype=np.float32)
Y = np.array(Y_list, dtype=np.float32)

# Quick data summary
print(f"  Speed range:  {X[:,19].min()*200:.1f} - {X[:,19].max()*200:.1f} km/h")
print(f"  Accel range:  {Y[:,0].min():.2f} - {Y[:,0].max():.2f}")
print(f"  Brake range:  {Y[:,1].min():.2f} - {Y[:,1].max():.2f}")
print(f"  Steer range:  {Y[:,2].min():.3f} - {Y[:,2].max():.3f}")


idx   = np.random.permutation(len(X))
n_val = int(len(X) * VAL_SPLIT)

X_train = torch.FloatTensor(X[idx[n_val:]])
Y_train = torch.FloatTensor(Y[idx[n_val:]])
X_val   = torch.FloatTensor(X[idx[:n_val]])
Y_val   = torch.FloatTensor(Y[idx[:n_val]])

print(f"Train: {len(X_train)}  Val: {len(X_val)}")

class RacingBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(22, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),                     nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return torch.tanh(self.network(x))

model     = RacingBrain()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def weighted_mse(pred, target):
    w = LOSS_WEIGHTS.to(pred.device)
    return ((pred - target) ** 2 * w).mean()

dataset = torch.utils.data.TensorDataset(X_train, Y_train)
loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"\nTraining for up to {EPOCHS} epochs...")
print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>9}")
print("-" * 34)

best_val = float('inf')
patience = 0

try:
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = weighted_mse(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(loader)

        model.eval()
        with torch.no_grad():
            val_loss = weighted_mse(model(X_val), Y_val).item()

        scheduler.step()

        if epoch % 50 == 0:
            marker = " <-- best" if val_loss < best_val else ""
            print(f"{epoch:>6} | {train_loss:>11.6f} | {val_loss:>9.6f}{marker}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUTPUT_PATH)
            patience = 0
        else:
            patience += 1
            if patience >= MAX_PATIENCE:
                print(f"\nEarly stop at epoch {epoch} (no improvement for {MAX_PATIENCE} epochs)")
                break

except KeyboardInterrupt:
    print("\nInterrupted — best model saved so far.")

print(f"\nBest val loss: {best_val:.6f}")
print(f"Saved: {OUTPUT_PATH}")

model.load_state_dict(torch.load(OUTPUT_PATH, weights_only=True))
model.eval()
with torch.no_grad():
    preds = model(X_val).numpy()

print(f"\nPrediction ranges (should not all be the same value):")
print(f"  accel: mean={preds[:,0].mean():.3f}  min={preds[:,0].min():.3f}  max={preds[:,0].max():.3f}")
print(f"  brake: mean={preds[:,1].mean():.3f}  min={preds[:,1].min():.3f}  max={preds[:,1].max():.3f}")
print(f"  steer: mean={preds[:,2].mean():.3f}  min={preds[:,2].min():.3f}  max={preds[:,2].max():.3f}")
print(f"\nReady. Run swarm_simple.py to start racing.")