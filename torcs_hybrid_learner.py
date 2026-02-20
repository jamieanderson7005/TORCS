import socket, sys, torch, os, numpy as np, time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PORT     = int(sys.argv[1]) if len(sys.argv) > 1 else 3001
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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


def corner_severity(track):
    """0=straight, 1=very tight corner"""
    front = float(min(track[7], track[8], track[9]))
    delta = float(abs(track[0] - track[18]))
    return float(np.clip(delta / 150.0 + (1.0 - front / 200.0), 0.0, 1.0))


def compute_reward(spd, tp, ang, track, prev_spd, damage=0.0):
    sev       = corner_severity(track)
    is_corner = sev > 0.25
    alignment = float(np.cos(ang))
    front_min = min(track[7:10])

    # Positive rewards
    speed_score = (spd / 250.0) * alignment * 1.6
    accel_bonus = max(0.0, (spd - prev_spd) * 0.12)

    # Stronger preference for center line (especially in corners)
    line_bonus = (1.0 - abs(tp)**1.6) * (0.9 if is_corner else 0.25)

    # PENALTIES
    # 1. Grass / kerb / near-off-track (very expensive)
    off_tarmac = 0.0
    if abs(tp) > 0.88:
        off_tarmac = -22.0 - (abs(tp) - 0.88) * 50.0

    # 2. Wall / barrier damage
    damage_cost = -damage * 0.12

    # 3. Excessive sliding
    slide_cost = 0.0
    if is_corner and abs(ang) > 0.18:
        slide_cost = -abs(ang) * 4.2
        if abs(ang) > 0.35:
            slide_cost -= 8.0

    # 4. Small penalty for being off-center on straights
    straight_dev = -abs(tp) * 0.15 if not is_corner else 0.0

    # 5. NEW: Hot corner entry
    hot_entry = 0.0
    if is_corner and spd > 140 and front_min < 80:
        hot_entry = (spd - 100) * 0.08

    total = (
        speed_score +
        accel_bonus +
        line_bonus +
        off_tarmac +
        damage_cost +
        slide_cost +
        straight_dev -
        hot_entry
    )

    return total


class HybridLearner:
    def __init__(self):
        self.model     = RacingBrain()
        self.save_path = os.path.join(BASE_DIR, f"brain_{PORT}_evolved.pth")
        self.time_path = os.path.join(BASE_DIR, f"laptime_{PORT}.txt")

        if os.path.exists(self.save_path):
            self.model.load_state_dict(
                torch.load(self.save_path, weights_only=True))
            print(f"[{PORT}] Loaded personal brain.")
        elif os.path.exists(os.path.join(BASE_DIR, "master_brain.pth")):
            self.model.load_state_dict(
                torch.load(os.path.join(BASE_DIR, "master_brain.pth"),
                           weights_only=True))
            print(f"[{PORT}] Loaded master brain.")
        else:
            print(f"[{PORT}] No brain found - starting fresh.")

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=5e-5, weight_decay=1e-5)

        self.so            = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.best_lap_time = float('inf')
        self.memory        = []
        self._load_best()

    def _load_best(self):
        if os.path.exists(self.time_path):
            try:
                self.best_lap_time = float(
                    open(self.time_path).read().strip())
                print(f"[{PORT}] Personal best: {self.best_lap_time:.2f}s")
            except:
                pass

    def _save_best(self, lap_time):
        torch.save(self.model.state_dict(), self.save_path)
        with open(self.time_path, "w") as f:
            f.write(str(lap_time))
        self.best_lap_time = lap_time
        print(f"[{PORT}] *** NEW BEST: {lap_time:.2f}s ***")

    def _update_brain(self):
        if len(self.memory) < 32:
            return

        # NEW: Filter out very bad samples (crash recoveries)
        filtered = []
        for m in self.memory:
            rew = m[2]
            tp_norm = m[0][20].item()  # norm_tp at index 20
            tp_val = tp_norm * 3.0     # denormalize
            if rew > -15 and abs(tp_val) < 1.8:
                filtered.append(m)

        if len(filtered) < 20:
            print(f"[{PORT}] Skipped update — too few good samples ({len(filtered)})")
            return

        states  = torch.stack([m[0] for m in filtered])
        actions = torch.stack([m[1] for m in filtered])

        self.optimizer.zero_grad()
        pred = self.model(states)
        loss = F.mse_loss(pred, actions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        print(f"[{PORT}] Brain updated — loss:{loss.item():.5f} | N:{len(filtered)}")
        self.memory = []

    def drive(self):
        INIT_MSG   = ("SCR(init -45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 "
                      ".5 1 1.7 2.5 4 7 12 19 45)").encode()
        ready_path = os.path.join(BASE_DIR, f"ready_{PORT}.flag")

        if os.path.exists(ready_path):
            os.remove(ready_path)

        self.so.settimeout(1.0)
        print(f"[{PORT}] Waiting for TORCS...")

        while True:
            try:
                self.so.sendto(INIT_MSG, ('localhost', PORT))
                data, _ = self.so.recvfrom(2**17)
                if b'identified' in data:
                    print(f"[{PORT}] Connected!")
                    with open(ready_path, 'w') as f:
                        f.write('ready')
                    break
            except socket.timeout:
                pass
            except ConnectionResetError:
                time.sleep(0.2)

        self.so.settimeout(None)

        current_gear  = 1
        last_shift    = 0.0
        prev_dist     = 0.0
        prev_spd      = 0.0
        episode_start = time.time()
        reward_sum    = 0.0
        steps         = 0
        lap_count     = 0

        best_str = ('--' if self.best_lap_time == float('inf')
                    else f'{self.best_lap_time:.2f}s')
        print(f"[{PORT}] Racing! Best: {best_str}")

        while True:
            try:
                data, _ = self.so.recvfrom(2**17)
                parts   = data.decode().strip().strip('()').split(')(')
                s       = {p.split(' ')[0]: p.split(' ')[1:] for p in parts}

                if 'track' not in s:
                    continue

                track = np.array(
                    [float(x) for x in s['track']], dtype=np.float32)
                spd   = float(s.get('speedX',   [0])[0])
                tp    = float(s.get('trackPos',  [0])[0])
                ang   = float(s.get('angle',     [0])[0])
                rpm   = float(s.get('rpm',       [0])[0])
                dist  = float(s.get('distRaced', [0])[0])
                damage = float(s.get('damage', [0])[0])

                # Lap detection
                if prev_dist > 500 and dist < 50:
                    lap_time   = time.time() - episode_start
                    avg_reward = reward_sum / max(steps, 1)
                    lap_count += 1
                    print(f"[{PORT}] Lap {lap_count}: {lap_time:.2f}s | "
                          f"reward={avg_reward:.3f} | best={self.best_lap_time:.2f}s")

                    self._update_brain()  # always try now (handles filtering)

                    if lap_time < self.best_lap_time:
                        self._save_best(lap_time)

                    episode_start = time.time()
                    reward_sum    = 0.0
                    steps         = 0

                prev_dist = dist

                # PROPER STATE NORMALIZATION
                norm_track = track / 200.0
                norm_spd   = spd / 300.0
                norm_tp    = tp / 3.0
                norm_ang   = ang / np.pi
                state = torch.tensor(
                    np.concatenate([norm_track, [norm_spd, norm_tp, norm_ang]]),
                    dtype=torch.float32)

                # POLICY + NOISE
                with torch.no_grad():
                    clean_action = self.model(state)

                sev = corner_severity(track)
                noise_scale = max(0.005, 0.04 - lap_count * 0.001)
                noise_scale *= max(0.25, 1.0 - sev * 0.75)  # even less noise in tight
                driven = torch.clamp(
                    clean_action + torch.randn(3) * noise_scale, -1.0, 1.0)

                accel = float(driven[0])
                brake = float(driven[1])
                steer = float(driven[2])

                front_min = min(track[7:10])

                # === AGGRESSIVE STEER CENTERING & BOOST ===
                steer_bias = -tp * 1.8
                if sev > 0.45:
                    steer = (steer + steer_bias) * (1.0 + sev * 1.4)
                else:
                    steer += steer_bias * 0.6
                steer = np.clip(steer, -1.0, 1.0)

                # === SAFETY NET (earlier & stronger) ===
                if tp > 0.75:
                    steer = max(steer, 0.55)
                elif tp < -0.75:
                    steer = min(steer, -0.55)

                # === ANTICIPATORY + EMERGENCY BRAKE ===
                if front_min < 120:
                    brake_boost = np.clip((120 - front_min) / 70.0, 0.0, 1.0) ** 1.5
                    brake_boost *= (1.0 + sev * 1.2)
                    brake = max(brake, brake_boost)
                    if front_min < 60:
                        accel = 0.0
                        brake = max(brake, 0.85 + (60 - front_min)/60.0)

                # === TIGHT CORNER OVERRIDE ===
                if front_min < 80:
                    print(f"[{PORT}] TIGHT! front={front_min:.1f}m sev={sev:.2f} tp={tp:.2f}")
                    steer = np.clip(tp * 3.5, -1.0, 1.0)
                    if front_min < 30:
                        steer = np.sign(tp) * -1.0   # full opposite lock

                # Mutual exclusion + straight safety
                throttle = max(0.0, accel - brake)
                brake    = max(0.0, brake - accel)
                if spd > 160 and abs(ang) < 0.12 and abs(tp) < 0.6:
                    brake = 0.0

                accel = float(np.clip(throttle, 0.0, 1.0))
                brake = float(np.clip(brake,    0.0, 1.0))
                steer = float(np.clip(steer,   -1.0, 1.0))

                # Reward
                rew = compute_reward(spd, tp, ang, track, prev_spd, damage)
                reward_sum += rew
                steps      += 1
                prev_spd    = spd

                self.memory.append((state, clean_action.detach(), rew))
                if len(self.memory) > 3000:
                    self.memory = self.memory[-3000:]

                # Gear
                now = time.time()
                if spd < 20:
                    if current_gear != 1:
                        current_gear = 1
                        last_shift   = now
                elif (now - last_shift) > 0.5:
                    if rpm > 6000 and current_gear < 6:
                        current_gear += 1
                        last_shift    = now
                    elif rpm < 3000 and current_gear > 1:
                        current_gear -= 1
                        last_shift    = now

                cmd = (f"(accel {accel:.3f})(brake {brake:.3f})"
                       f"(steer {steer:.4f})(gear {current_gear})")
                self.so.sendto(cmd.encode(), ('localhost', PORT))

            except Exception as e:
                if isinstance(e, ConnectionResetError):
                    time.sleep(0.1)
                continue


if __name__ == "__main__":
    HybridLearner().drive()