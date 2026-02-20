import socket, sys, os, time, csv, subprocess, random
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "base_racing_line.csv")
TORCS_ROOT = r"C:\Users\User\Downloads\torcs\torcs"
TORCS_EXE  = os.path.join(TORCS_ROOT, "wtorcs.exe")
PORT       = 3001

STEER_GAIN       = 11.0
TP_CENTER_GAIN   = 0.95          # ← SUPER LOOSE → car stays on biased path!
MAX_BRAKE        = 0.65
JITTER_PROB      = 0.04
JITTER_MAG       = 0.05

SPEED_SCALE_RANGE       = (2.0, 8.2)
CURVE_PENALTY_RANGE     = (0.7, 4.8)
MIN_CORNER_SPEED_RANGE  = (52.0, 110.0)
BRAKE_SENS_RANGE        = (8.0, 20.0)

# MAXIMUM bias range!
TRACKPOS_BIAS_RANGE     = (-1.00, 1.00)

TARGET_SPD_ALPHA  = 0.22
SPEED_DEADBAND    = 3.5
BRAKE_HYST        = 1.8
UPSHIFT_RPM       = 7800
DOWNSHIFT_RPM     = 3200


def new_lap_style():
    extreme = random.random() < 0.50  # 50% extremes!

    if extreme:
        style_type = random.choice(['late_brake', 'early_brake', 'tight_line', 'wide_line', 'max_speed'])
        s = dict(
            speed_scale       = 4.2,
            curve_penalty     = 2.2,
            min_corner_speed  = 78.0,
            brake_sensitivity = 12.0,
            trackpos_bias     = 0.0,
        )
        desc_parts = []

        if style_type == 'late_brake':
            s['brake_sensitivity'] = 18.0
            s['speed_scale']       = 7.6
            s['trackpos_bias']     = random.uniform(-0.30, 0.30)
            desc_parts.append("late braking")

        elif style_type == 'early_brake':
            s['brake_sensitivity'] = 8.5
            s['min_corner_speed']  = 55.0
            s['trackpos_bias']     = random.uniform(0.20, 0.50)
            desc_parts.append("early braking")

        elif style_type == 'tight_line':
            s['trackpos_bias']  = random.uniform(-0.98, -0.55)  # ← EXTREME LEFT!
            s['curve_penalty']  = 0.9
            desc_parts.append("*** EXTREME TIGHT/LEFT ***")

        elif style_type == 'wide_line':
            s['trackpos_bias']  = random.uniform(0.55, 0.98)    # ← EXTREME RIGHT!
            s['speed_scale']    = 7.1
            desc_parts.append("*** EXTREME WIDE/RIGHT ***")

        else:  # max_speed
            s['min_corner_speed'] = 105.0
            s['speed_scale']      = 8.0
            s['curve_penalty']    = 0.8
            s['trackpos_bias']    = random.uniform(-0.40, 0.40)
            desc_parts.append("max corner speed")

        desc = "EXTREME: " + " | ".join(desc_parts)

    else:
        s = dict(
            speed_scale       = random.uniform(*SPEED_SCALE_RANGE),
            curve_penalty     = random.uniform(*CURVE_PENALTY_RANGE),
            min_corner_speed  = random.uniform(*MIN_CORNER_SPEED_RANGE),
            brake_sensitivity = random.uniform(*BRAKE_SENS_RANGE),
            trackpos_bias     = random.uniform(*TRACKPOS_BIAS_RANGE),
        )
        desc = (f"spd={s['speed_scale']:.1f} | crv={s['curve_penalty']:.1f} | "
                f"min={s['min_corner_speed']:.0f} | brk={s['brake_sensitivity']:.1f} | "
                f"BIAS={s['trackpos_bias']:.2f} ← WATCH THIS!")

    return s, desc


def compute_target_speed(track, style):
    front_center = float(min(track[6:11]))
    curve_delta = max(
        abs(track[0] - track[18]),
        abs(track[2] - track[16]),
        abs(track[4] - track[14]),
        abs(track[6] - track[12])
    )
    raw = (front_center * style['speed_scale']) \
          - (curve_delta * style['curve_penalty'] * 1.18)
    return max(style['min_corner_speed'], raw)


def compute_controls(track, spd, tp, ang, rpm, gear, last_shift, style, prev_target_spd=80.0):
    raw_target = compute_target_speed(track, style)
    target_spd = (TARGET_SPD_ALPHA * raw_target) + ((1 - TARGET_SPD_ALPHA) * prev_target_spd)

    desired_tp = style['trackpos_bias']
    tp_error   = tp - desired_tp

    steer_rec = np.clip(
        (ang * STEER_GAIN / np.pi)
        - (tp_error * TP_CENTER_GAIN),
        -1.0, 1.0)

    jitter = (random.uniform(-JITTER_MAG, JITTER_MAG) if random.random() < JITTER_PROB else 0.0)
    steer_app = np.clip(steer_rec + jitter, -1.0, 1.0)

    speed_error = spd - target_spd
    if spd < 30:
        accel, brake = 1.0, 0.0
    elif abs(speed_error) < SPEED_DEADBAND:
        accel, brake = 0.4, 0.0
    elif speed_error > BRAKE_HYST:
        brake_amount = np.clip((speed_error - BRAKE_HYST) / style['brake_sensitivity'], 0.0, MAX_BRAKE)
        brake_amount *= (1.0 - abs(steer_rec) * 0.75)
        accel, brake = 0.0, brake_amount
    else:
        accel, brake = 1.0, 0.0

    now = time.time()
    shift_ready = (now - last_shift) > 0.5
    if spd < 20:
        if gear != 1:
            gear = 1
            last_shift = now
    elif shift_ready:
        if rpm > UPSHIFT_RPM and gear < 6:
            gear += 1
            last_shift = now
        elif rpm < DOWNSHIFT_RPM and gear > 1:
            gear -= 1
            last_shift = now

    return accel, brake, steer_rec, steer_app, gear, last_shift, target_spd


def destringify(s):
    if not s: return s
    if isinstance(s, str):
        try: return float(s)
        except: return s
    elif isinstance(s, list):
        return [destringify(i) for i in s] if len(s) > 1 else destringify(s[0])
    return s


def parse_telemetry(raw):
    try:
        parts = raw.decode().strip().strip('()').split(')(')
        return {p.split(' ')[0]: destringify(p.split(' ')[1:]) for p in parts}
    except:
        return {}


if __name__ == "__main__":
    print("Starting TORCS...")
    subprocess.Popen([TORCS_EXE, "-p", str(PORT)], cwd=TORCS_ROOT)
    time.sleep(3)

    so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    so.settimeout(2.0)

    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ([f'track_{i}' for i in range(19)] + ['speedX', 'trackPos', 'angle', 'accel', 'brake', 'steer'])
        writer.writerow(headers)

    csv_file = open(CSV_PATH, 'a', newline='')
    writer = csv.writer(csv_file)

    current_gear  = 1
    last_shift    = 0.0
    frames_saved  = 0
    identified    = False

    prev_laps     = 0
    lap_count     = 0
    prev_target_spd = 80.0

    style, desc = new_lap_style()
    print(f"\nLap 1 style: {desc}")

    print(f"Connecting to TORCS on port {PORT}...")
    print("In TORCS: Race → Quick Race → Start.")
    print("WATCH CONSOLE: Different BIAS = different paths! tp should ≈ BIAS.")

    try:
        while True:
            try:
                so.sendto("SCR(init -45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45)".encode(), ('localhost', PORT))
                d, _ = so.recvfrom(2**17)
                if b'identified' in d:
                    print("Connected! MAX variation + laps sensor... Ctrl+C to stop.")
                    identified = True
            except (socket.timeout, ConnectionResetError):
                continue

            if not identified:
                continue

            while True:
                try:
                    d, _ = so.recvfrom(2**17)
                    s = parse_telemetry(d)
                    if 'track' not in s:
                        continue

                    track = [float(x) for x in (s['track'] if isinstance(s['track'], list) else [s['track']]*19)]
                    spd   = float(s.get('speedX', 0))
                    tp    = float(s.get('trackPos', 0))
                    ang   = float(s.get('angle', 0))
                    rpm   = float(s.get('rpm', 0))
                    laps  = int(s.get('laps', prev_laps))

                    # RELIABLE LAP DETECTION using 'laps' sensor!
                    if laps > prev_laps:
                        lap_count = laps
                        style, desc = new_lap_style()
                        print(f"\n🎉 NEW LAP {lap_count} (laps={laps}) → {desc}")
                        print(f"     TARGET BIAS={style['trackpos_bias']:.3f} ← CAR SHOULD HUG THIS!")
                        prev_target_spd = 80.0

                    prev_laps = laps

                    accel, brake, steer_rec, steer_app, current_gear, last_shift, prev_target_spd = compute_controls(
                        track, spd, tp, ang, rpm, current_gear, last_shift, style, prev_target_spd)

                    # RELAXED filter to CAPTURE extreme lines
                    if spd > 5 and abs(tp) < 1.4:
                        writer.writerow(track + [spd, tp, ang, accel, brake, steer_rec])
                        csv_file.flush()
                        frames_saved += 1
                        if frames_saved % 800 == 0:  # more frequent debug
                            print(f"  {frames_saved} frames | spd={spd:.0f} tp={tp:.3f} (bias={style['trackpos_bias']:.3f}) rpm={rpm:.0f} | laps={laps}")

                    cmd = f"(accel {accel:.3f})(brake {brake:.3f})(steer {steer_app:.4f})(gear {current_gear})"
                    so.sendto(cmd.encode(), ('localhost', PORT))

                except socket.timeout:
                    print("Connection lost, re-initialising...")
                    break
                except ConnectionResetError:
                    time.sleep(0.1)
                    continue

    except KeyboardInterrupt:
        csv_file.close()
        print(f"\nDone. {frames_saved} frames across {lap_count} laps")
        print(f"Saved to {CSV_PATH}")
        print("Run train_model.py next.")