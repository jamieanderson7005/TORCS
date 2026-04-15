#Req libs

import json
import os
import time
import random
import sys
import json as _json
import numpy as np
from dataclasses import dataclass, field
from typing import List


def _extract_mode() -> str: #Def how to run the program through the modes.
    mode     = "continuous"
    new_argv = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg == "--drive":        mode = "drive"
        elif arg == "--optimise":   mode = "optimise"
        elif arg == "--continuous": mode = "continuous"
        else: new_argv.append(arg)
    sys.argv = new_argv
    return mode

_MODE = _extract_mode() #Saves the mode

# Constants | Can be adjusted

PI = 3.14159265359

POPULATION_SIZE   = 10
NUM_PARENTS       = 1
NUM_OFFSPRING     = 10
NUM_GENERATIONS   = 30
EPISODE_MAX_STEPS = 15000  

MUTATION_SIGMA    = 0.06
SIGMA_DECAY       = 0.95
SIGMA_FLOOR       = 0.02

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimisation_results.json") #Saves the results for either more trianing or loading.
LAP_LENGTH_M      = 3600.0

# Fitness constants
LAP_TIME_SCALE    = 100_000.0
DISTANCE_SCALE    = 0.25
OFF_TRACK_PENALTY = 2.0
CRASH_SCALE       = 15.0
OFF_TRACK_THRESHOLD = 0.98
COMPLETION_BONUS  = 300.0   

@dataclass
class DriveParams: #Defines the parameters for the driving behaviour, these are the base before being optimised
    straight_speed:   float = 80.0    
    corner_speed_min: float = 40.0    
    corner_lookahead: float = 120.0   
    brake_lookahead:  float = 200.0   
    steer_angle_gain:        float = 22.0
    steer_center_gain:       float = 0.25
    steer_speed_correction:  float = 4.0
    steer_anticipation_gain: float = 0.15  
    accel_increment: float = 0.4
    accel_decrement: float = 0.4
    brake_pressure:    float = 0.9
    brake_speed_range: float = 25.0
    tcs_slip_threshold: float = 2.0
    tcs_accel_cut:      float = 0.15
    racing_line_gain:  float = 0.80
    racing_line_blend: float = 0.55   
    gear_speeds: List[float] = field(
        default_factory=lambda: [90.0, 150.0, 200.0, 250.0, 300.0]
    )
    def to_vector(self) -> list:
        return [
            self.straight_speed,
            self.corner_speed_min,
            self.corner_lookahead,
            self.brake_lookahead,
            self.steer_angle_gain,
            self.steer_center_gain,
            self.steer_speed_correction,
            self.steer_anticipation_gain,
            self.accel_increment,
            self.accel_decrement,
            self.brake_pressure,
            self.brake_speed_range,
            self.tcs_slip_threshold,
            self.tcs_accel_cut,
            self.racing_line_gain,
            self.racing_line_blend,
        ] + list(self.gear_speeds)

    @classmethod
    def from_vector(cls, v: list) -> "DriveParams":
        return cls(
            straight_speed          = v[0],
            corner_speed_min        = v[1],
            corner_lookahead        = v[2],
            brake_lookahead         = v[3],
            steer_angle_gain        = v[4],
            steer_center_gain       = v[5],
            steer_speed_correction  = v[6],
            steer_anticipation_gain = v[7],
            accel_increment         = v[8],
            accel_decrement         = v[9],
            brake_pressure          = v[10],
            brake_speed_range       = v[11],
            tcs_slip_threshold      = v[12],
            tcs_accel_cut           = v[13],
            racing_line_gain        = v[14],
            racing_line_blend       = v[15],
            gear_speeds             = list(v[16:21]),
        )

    @staticmethod
    def n_params() -> int:
        return 21

    @staticmethod
    def bounds() -> dict:
        return {
            "straight_speed":         (80.0,  340.0),
            "corner_speed_min":       (35.0,   72.0),
            "corner_lookahead":       (80.0,  200.0),
            "brake_lookahead":        (120.0, 350.0), 
            "steer_angle_gain":       (5.0,    50.0),
            "steer_center_gain":      (0.0,    0.6),
            "steer_speed_correction": (0.5,   15.0),
            "steer_anticipation_gain":(0.0,    0.5),
            "accel_increment":        (0.05,   1.0),
            "accel_decrement":        (0.05,   1.0),
            "brake_pressure":         (0.3,    1.0),
            "brake_speed_range":      (5.0,   80.0),
            "tcs_slip_threshold":     (0.3,    8.0),
            "tcs_accel_cut":          (0.01,   0.5),
            "racing_line_gain":       (0.0,    1.0),
            "racing_line_blend":      (0.0,    1.0),
            "gear_speed_2":           (70.0,  120.0),
            "gear_speed_3":           (120.0, 180.0),
            "gear_speed_4":           (170.0, 230.0),
            "gear_speed_5":           (220.0, 280.0),
            "gear_speed_6":           (270.0, 330.0),
        }

    def to_dict(self) -> dict:
        return {
            "straight_speed":         self.straight_speed,
            "corner_speed_min":       self.corner_speed_min,
            "corner_lookahead":       self.corner_lookahead,
            "brake_lookahead":        self.brake_lookahead,
            "steer_angle_gain":       self.steer_angle_gain,
            "steer_center_gain":      self.steer_center_gain,
            "steer_speed_correction": self.steer_speed_correction,
            "steer_anticipation_gain":self.steer_anticipation_gain,
            "accel_increment":        self.accel_increment,
            "accel_decrement":        self.accel_decrement,
            "brake_pressure":         self.brake_pressure,
            "brake_speed_range":      self.brake_speed_range,
            "tcs_slip_threshold":     self.tcs_slip_threshold,
            "tcs_accel_cut":          self.tcs_accel_cut,
            "racing_line_gain":       self.racing_line_gain,
            "racing_line_blend":      self.racing_line_blend,
            "gear_speeds":            self.gear_speeds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DriveParams":
        straight = d.get("straight_speed", d.get("target_speed", 80.0))
        return cls(
            straight_speed         = straight,
            corner_speed_min       = d.get("corner_speed_min",   40.0),
            corner_lookahead       = d.get("corner_lookahead",  120.0),
            brake_lookahead        = d.get("brake_lookahead",   200.0),
            steer_angle_gain        = d["steer_angle_gain"],
            steer_center_gain       = d["steer_center_gain"],
            steer_speed_correction  = d["steer_speed_correction"],
            steer_anticipation_gain = d.get("steer_anticipation_gain", 0.15),
            accel_increment        = d["accel_increment"],
            accel_decrement        = d["accel_decrement"],
            brake_pressure         = d["brake_pressure"],
            brake_speed_range      = d["brake_speed_range"],
            tcs_slip_threshold     = d["tcs_slip_threshold"],
            tcs_accel_cut          = d["tcs_accel_cut"],
            racing_line_gain       = d.get("racing_line_gain",  0.80),
            racing_line_blend      = d.get("racing_line_blend", 0.55),
            gear_speeds            = d["gear_speeds"],
        )


_FWD_IDX = [7, 8, 9, 10, 11]
_CURVE_LEFT_IDX  = [5, 6]
_CURVE_RIGHT_IDX = [12, 13]


def _corner_curvature(track: list) -> float:
    if len(track) < 14:
        return 0.0
    left_fwd  = min(track[i] for i in _CURVE_LEFT_IDX)
    right_fwd = min(track[i] for i in _CURVE_RIGHT_IDX)
    total     = left_fwd + right_fwd + 1e-6
    return float(np.clip((left_fwd - right_fwd) / total, -1.0, 1.0))


def _corner_tightness(track: list, lookahead: float) -> float:
    if len(track) < 12:
        return 0.0
    fwd = min(track[i] for i in _FWD_IDX)
    return float(np.clip(1.0 - fwd / max(lookahead, 1.0), 0.0, 1.0))

_CORKSCREW_ENTRY = 2330.0
_CORKSCREW_EXIT  = 2520.0

def _in_corkscrew(dist_from_start: float) -> bool:
    return _CORKSCREW_ENTRY <= dist_from_start <= _CORKSCREW_EXIT

def _corkscrew_speed_cap(dist_from_start: float, corner_speed_min: float) -> float:
    cap = corner_speed_min + 20.0
    ramp = 100.0
    if dist_from_start < _CORKSCREW_ENTRY:
        t = max(0.0, (dist_from_start - (_CORKSCREW_ENTRY - ramp)) / ramp)
        return cap + (1.0 - t) * 9999.0
    elif dist_from_start > _CORKSCREW_EXIT:
        t = min(1.0, (dist_from_start - _CORKSCREW_EXIT) / ramp)
        return cap + t * 9999.0
    else:
        return cap


_LASTCORNER_ENTRY = 3190.0
_LASTCORNER_EXIT  = 3300.0

def _in_lastcorner(dist_from_start: float) -> bool:
    return _LASTCORNER_ENTRY <= dist_from_start <= _LASTCORNER_EXIT

def _lastcorner_speed_cap(dist_from_start: float, corner_speed_min: float) -> float:
    cap  = corner_speed_min + 5.0
    ramp = 150.0
    if dist_from_start < _LASTCORNER_ENTRY:
        t = max(0.0, (dist_from_start - (_LASTCORNER_ENTRY - ramp)) / ramp)
        return cap + (1.0 - t) * 9999.0
    elif dist_from_start > _LASTCORNER_EXIT:
        t = min(1.0, (dist_from_start - _LASTCORNER_EXIT) / ramp)
        return cap + t * 9999.0
    else:
        return cap


def drive(c, params: DriveParams = None):
    if params is None:
        params = DriveParams()

    S, R = c.S.d, c.R.d

    speed     = S.get('speedX',   0.0)
    angle     = S.get('angle',    0.0)
    track_pos = S.get('trackPos', 0.0)
    track     = S.get('track',    [])

    if abs(track_pos) > 1.0:
        R['steer'] = float(np.clip(-track_pos * 0.9, -1.0, 1.0))
        R['brake'] = 0.5
        R['accel'] = 0.0
        R['gear']  = 1
        return

    tightness   = _corner_tightness(track, params.corner_lookahead)
    brake_tight = _corner_tightness(track, params.brake_lookahead)
    curvature   = _corner_curvature(track)
    on_approach = brake_tight > 0.05
    at_apex     = tightness >= 0.45
    on_exit     = (tightness < 0.08) and not on_approach

    if on_exit or not on_approach:
        target_pos = 0.0
    elif at_apex:
        apex_t     = curvature * params.racing_line_gain * params.racing_line_blend * 0.45
        target_pos = float(np.clip(apex_t, -0.55, 0.55))
    else:
        approach_factor = float(np.clip(brake_tight * 3.0, 0.0, 1.0))
        wide_t          = -curvature * params.racing_line_gain * approach_factor
        target_pos      = float(np.clip(wide_t, -0.82, 0.82))

    steer_angle  = angle * params.steer_angle_gain / PI

    pos_error    = track_pos - target_pos
    pos_boost    = 1.0 + brake_tight * 5.0
    steer_pos    = -pos_error * params.steer_center_gain * pos_boost

    anticipation = curvature * tightness * params.steer_anticipation_gain

    R['steer'] = float(np.clip(steer_angle + steer_pos + anticipation, -1.0, 1.0))

    speed_range      = max(0.0, params.straight_speed - params.corner_speed_min)
    corner_target    = params.straight_speed - tightness * speed_range
    steer_penalty    = abs(R['steer']) * params.steer_speed_correction
    effective_target = max(params.corner_speed_min, corner_target - steer_penalty)

    # zone override
    dist_from_start  = S.get('distFromStart', -1.0)
    in_lastcorner    = False

    if dist_from_start >= 0:
        if _in_corkscrew(dist_from_start):
            effective_target = min(effective_target,
                _corkscrew_speed_cap(dist_from_start, params.corner_speed_min))
        if _in_lastcorner(dist_from_start):
            effective_target = min(effective_target,
                _lastcorner_speed_cap(dist_from_start, params.corner_speed_min))
            in_lastcorner = True

    if on_exit or not (brake_tight > 0.05):
        R['brake'] = 0.0
        R['accel'] = min(1.0, R['accel'] + params.accel_increment)
    elif speed > effective_target:
        R['accel'] = max(0.0, R['accel'] - params.accel_decrement)
        overspeed  = speed - effective_target
        R['brake'] = float(np.clip(
            params.brake_pressure * overspeed / max(params.brake_speed_range, 1.0),
            0.0, params.brake_pressure
        ))
    else:
        R['accel'] = min(1.0, R['accel'] + params.accel_increment)
        R['brake'] = 0.0
    if in_lastcorner:
        R['accel'] = min(R['accel'], 0.15)
        R['steer'] = float(np.clip(R['steer'], -0.4, 0.4))

    R['accel'] = float(np.clip(R['accel'], 0.0, 1.0))
    R['brake'] = float(np.clip(R['brake'], 0.0, 1.0))
    rear_spin  = S['wheelSpinVel'][2] + S['wheelSpinVel'][3]
    front_spin = S['wheelSpinVel'][0] + S['wheelSpinVel'][1]
    if (rear_spin - front_spin) > params.tcs_slip_threshold:
        R['accel'] = max(0.0, R['accel'] - params.tcs_accel_cut)

    gs        = params.gear_speeds
    R['gear'] = 1
    if speed > gs[0]: R['gear'] = 2
    if speed > gs[1]: R['gear'] = 3
    if speed > gs[2]: R['gear'] = 4
    if speed > gs[3]: R['gear'] = 5
    if speed > gs[4]: R['gear'] = 6


BOUNDS     = DriveParams.bounds()
BOUND_KEYS = list(BOUNDS.keys())
LOW        = np.array([BOUNDS[k][0] for k in BOUND_KEYS])
HIGH       = np.array([BOUNDS[k][1] for k in BOUND_KEYS])

assert len(LOW) == DriveParams.n_params(), \
    f"bounds dict has {len(LOW)} keys but to_vector() has {DriveParams.n_params()} values"


def _clip(v):
    return np.clip(v, LOW, HIGH)


def _sanitise(p: DriveParams) -> DriveParams:
    p.gear_speeds      = sorted(p.gear_speeds)
    p.corner_speed_min = min(p.corner_speed_min, p.straight_speed)
    p.brake_lookahead  = max(p.brake_lookahead, p.corner_lookahead)
    return p


def _random_individual():
    return LOW + np.random.rand(len(LOW)) * (HIGH - LOW)


def _mutate(parent: np.ndarray, sigma: float) -> np.ndarray:
    scale = (HIGH - LOW) * sigma
    return _clip(parent + np.random.randn(len(parent)) * scale)

class _Client:

    class _State:
        def __init__(self):
            self.d = {
                'speedX': 0.0, 'speedY': 0.0, 'speedZ': 0.0,
                'angle': 0.0, 'trackPos': 0.0,
                'wheelSpinVel': [0.0, 0.0, 0.0, 0.0],
                'track': [200.0] * 19,
                'rpm': 0.0,
                'opponents': [200.0] * 36,
                'distFromStart': 0.0,
                'distRaced': 0.0,
                'damage': 0.0,
                'fuel': 0.0,
                'wheelSkid': [0.0, 0.0, 0.0, 0.0],
                'lastLapTime': 0.0,
                'curLapTime': 0.0,
                'racePos': 0,
            }

    class _Response:
        def __init__(self):
            self.d = {'accel': 0.2, 'brake': 0.0, 'gear': 1,
                      'steer': 0.0, 'clutch': 0.0, 'meta': 0}

    def __init__(self):
        self.S = self._State()
        self.R = self._Response()

    def update(self, obs, env):
        spd = env.default_speed
        self.S.d['speedX']       = float(obs.speedX) * spd
        self.S.d['speedY']       = float(obs.speedY) * spd
        self.S.d['speedZ']       = float(obs.speedZ) * spd
        self.S.d['wheelSpinVel'] = list(obs.wheelSpinVel.astype(float))
        self.S.d['track']        = list(obs.track.astype(float) * 200.0)
        self.S.d['rpm']          = float(obs.rpm)
        self.S.d['opponents']    = list(obs.opponents.astype(float) * 200.0)
        try:
            raw = env.client.S.d
            for key in ('angle', 'trackPos', 'distFromStart', 'distRaced',
                        'damage', 'fuel', 'lastLapTime', 'curLapTime', 'racePos'):
                self.S.d[key] = float(raw.get(key, 0.0))
            if 'wheelSkid' in raw:
                self.S.d['wheelSkid'] = list(raw['wheelSkid'])
        except Exception:
            pass

    def to_action(self, env) -> np.ndarray:
        steer = float(np.clip(self.R.d['steer'], -1.0, 1.0))
        accel = float(np.clip(self.R.d.get('accel', 0.0), 0.0, 1.0))
        brake = float(np.clip(self.R.d.get('brake', 0.0), 0.0, 1.0))
        gear  = int(self.R.d.get('gear', 1))
        return np.array([steer, accel, brake, gear], dtype=np.float32)

def _evaluate(env, params: DriveParams, max_steps: int,
              relaunch: bool = False) -> dict:
    obs    = env.reset(relaunch=relaunch)
    client = _Client()
    client.update(obs, env)

    lap_start_step  = None
    lap_time        = None
    dist_raced      = 0.0
    off_track_steps = 0

    for step in range(max_steps):
        client.update(obs, env)
        drive(client, params)
        action = client.to_action(env)
        obs, reward, done, _ = env.step(action)

        dist_raced = max(client.S.d.get('distRaced', 0.0), dist_raced)
        track_pos  = client.S.d['trackPos']

        if lap_start_step is None and dist_raced > 2.0:
            lap_start_step = step

        if abs(track_pos) > OFF_TRACK_THRESHOLD:
            off_track_steps += 1

        if lap_start_step is not None and dist_raced >= LAP_LENGTH_M * 0.98:
            lap_time = (step - lap_start_step) / 50.0
            done     = True

        if done:
            break

    dist_raced = max(dist_raced, 0.0)
    fraction   = min(1.0, dist_raced / LAP_LENGTH_M)
    if lap_time is not None:
        lap_score  = (LAP_TIME_SCALE / max(lap_time, 1.0)) ** 3 / 100_000.0
        dist_score = dist_raced * DISTANCE_SCALE
        crash_pen  = 0.0
        completion = COMPLETION_BONUS
    else:
        lap_score  = 0.0
        dist_score = dist_raced * DISTANCE_SCALE
        crash_pen  = CRASH_SCALE * (1.0 - fraction)
        completion = 0.0

    off_pen  = off_track_steps * OFF_TRACK_PENALTY
    fitness  = lap_score + dist_score + completion - off_pen - crash_pen

    return {
        "fitness":         fitness,
        "lap_time":        lap_time,
        "dist_raced":      dist_raced,
        "off_track_steps": off_track_steps,
    }

def run_optimisation_round(
    env,
    seed_params:  DriveParams,
    seed_fitness: float,
    history:      list,
    round_number: int,
    sigma:        float,
) -> tuple:
    print(f"\n{'=' * 65}")
    print(f"  Round {round_number}   seed_fitness={seed_fitness:.1f}"
          f"   sigma={sigma:.4f}")
    print(f"  Pop={POPULATION_SIZE}  Parents={NUM_PARENTS}"
          f"  Offspring={NUM_OFFSPRING}  Gens={NUM_GENERATIONS}")
    print(f"{'=' * 65}")

    population    = [_random_individual() for _ in range(POPULATION_SIZE)]
    population[0] = _clip(np.array(seed_params.to_vector()))

    best_fitness    = seed_fitness
    best_params     = seed_params
    episode_counter = 0

    for gen in range(NUM_GENERATIONS):
        print(f"\n── Gen {gen+1}/{NUM_GENERATIONS}  sigma={sigma:.4f} ──")

        fitnesses = []
        for idx, vec in enumerate(population):
            params   = _sanitise(DriveParams.from_vector(vec.tolist()))
            relaunch = (episode_counter % 6 == 0)
            episode_counter += 1

            result = _evaluate(env, params, EPISODE_MAX_STEPS,
                               relaunch=relaunch)
            f = result["fitness"]
            fitnesses.append(f)

            lt  = f"{result['lap_time']:.1f}s" if result['lap_time'] else "DNF"
            tag = "  <- seed" if (idx == 0 and gen == 0) else ""
            print(f"  [{idx+1:2d}]  fit={f:8.1f}  lap={lt:7s}"
                  f"  dist={result['dist_raced']:5.0f}m"
                  f"  vS={params.straight_speed:.0f}"
                  f"  vC={params.corner_speed_min:.0f}{tag}")

        ranked     = sorted(zip(fitnesses, population),
                            key=lambda x: x[0], reverse=True)
        parents    = [v for _, v in ranked[:NUM_PARENTS]]
        best_f, best_v = ranked[0]

        print(f"  -> Best this gen: {best_f:.1f}")

        if best_f > best_fitness:
            best_fitness = best_f
            best_params  = _sanitise(DriveParams.from_vector(best_v.tolist()))
            print("  *** New overall best — saving ***")
            save_results(best_params, best_fitness, history)

        history.append({
            "round":      round_number,
            "generation": gen + 1,
            "best":       best_f,
            "mean":       float(np.mean(fitnesses)),
            "sigma":      sigma,
        })

        offspring  = [_mutate(random.choice(parents), sigma)
                      for _ in range(NUM_OFFSPRING)]
        population = parents + offspring
        sigma      = max(sigma * SIGMA_DECAY, SIGMA_FLOOR)

    return best_params, best_fitness, history, sigma


def run_continuous_optimisation(
    env,
    seed_params:  DriveParams,
    seed_fitness: float,
    history:      list,
    start_gen:    int,
    sigma:        float,
):

    print(f"\n{'=' * 65}")
    print(f"  Continuous mode — runs until Ctrl-C")
    print(f"  seed_fitness={seed_fitness:.1f}   sigma={sigma:.4f}")
    print(f"  Pop={POPULATION_SIZE}  Parents={NUM_PARENTS}"
          f"  Offspring={NUM_OFFSPRING}")
    print(f"{'=' * 65}")

    population    = [_random_individual() for _ in range(POPULATION_SIZE)]
    population[0] = _clip(np.array(seed_params.to_vector()))

    best_fitness    = seed_fitness
    best_params     = seed_params
    episode_counter = 0
    gen             = start_gen

    while True:
        gen += 1
        print(f"\n── Gen {gen}  sigma={sigma:.4f} ──")

        fitnesses = []
        for idx, vec in enumerate(population):
            params   = _sanitise(DriveParams.from_vector(vec.tolist()))
            relaunch = (episode_counter % 6 == 0)
            episode_counter += 1

            result = _evaluate(env, params, EPISODE_MAX_STEPS,
                               relaunch=relaunch)
            f = result["fitness"]
            fitnesses.append(f)

            lt = f"{result['lap_time']:.1f}s" if result['lap_time'] else "DNF"
            print(f"  [{idx+1:2d}]  fit={f:8.1f}  lap={lt:7s}"
                  f"  dist={result['dist_raced']:5.0f}m"
                  f"  vS={params.straight_speed:.0f}"
                  f"  vC={params.corner_speed_min:.0f}")

        ranked     = sorted(zip(fitnesses, population),
                            key=lambda x: x[0], reverse=True)
        parents    = [v for _, v in ranked[:NUM_PARENTS]]
        best_f, best_v = ranked[0]

        print(f"  -> Best this gen: {best_f:.1f}  (overall best: {best_fitness:.1f})")

        if best_f > best_fitness:
            best_fitness = best_f
            best_params  = _sanitise(DriveParams.from_vector(best_v.tolist()))
            print("  *** New overall best — saving ***")
            save_results(best_params, best_fitness, history)

        history.append({
            "generation": gen,
            "best":       best_f,
            "mean":       float(np.mean(fitnesses)),
            "sigma":      sigma,
        })

        offspring  = [_mutate(random.choice(parents), sigma)
                      for _ in range(NUM_OFFSPRING)]
        population = parents + offspring

        new_sigma = sigma * SIGMA_DECAY
        if new_sigma < SIGMA_FLOOR:

            print(f"  [sigma floor reached — resetting to {MUTATION_SIGMA}]")
            sigma = MUTATION_SIGMA
        else:
            sigma = new_sigma

def save_results(params: DriveParams, fitness: float, history: list):
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "best_fitness": fitness,
            "best_params":  params.to_dict(),
            "history":      history,
        }, f, indent=2)
    print(f"  [saved to {RESULTS_FILE}]")


def load_best_params():
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    return (DriveParams.from_dict(data["best_params"]),
            data.get("best_fitness", -np.inf),
            data.get("history", []))


def _print_winner(params: DriveParams, fitness: float):
    print("=" * 65)
    print(f"  Best fitness: {fitness:.1f}")
    print(f"""
  DriveParams(
    straight_speed         = {params.straight_speed:.1f},
    corner_speed_min       = {params.corner_speed_min:.1f},
    corner_lookahead       = {params.corner_lookahead:.1f},
    brake_lookahead        = {params.brake_lookahead:.1f},
    steer_angle_gain       = {params.steer_angle_gain:.4f},
    steer_center_gain      = {params.steer_center_gain:.4f},
    steer_speed_correction = {params.steer_speed_correction:.4f},
    steer_anticipation_gain= {params.steer_anticipation_gain:.4f},
    accel_increment        = {params.accel_increment:.4f},
    accel_decrement        = {params.accel_decrement:.4f},
    brake_pressure         = {params.brake_pressure:.4f},
    brake_speed_range      = {params.brake_speed_range:.4f},
    tcs_slip_threshold     = {params.tcs_slip_threshold:.4f},
    tcs_accel_cut          = {params.tcs_accel_cut:.4f},
    racing_line_gain       = {params.racing_line_gain:.4f},
    racing_line_blend      = {params.racing_line_blend:.4f},
    gear_speeds            = {[round(g, 1) for g in params.gear_speeds]},
  )""")
    print("=" * 65)

TELEMETRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torcs_telemetry.json")

_tyre_skid_accum = [0.0, 0.0, 0.0, 0.0]
_session_start_time = None

def _write_telemetry(S, R):
    global _tyre_skid_accum, _session_start_time

    if _session_start_time is None:
        _session_start_time = time.time()

    skid = S.get('wheelSkid', [0.0, 0.0, 0.0, 0.0])
    for i in range(4):
        _tyre_skid_accum[i] += abs(skid[i]) if i < len(skid) else 0.0

    wear_scale = 5000.0
    tyre_wear  = [min(100.0, round(a / wear_scale * 100.0, 1))
                  for a in _tyre_skid_accum]

    cur_lap  = S.get('curLapTime',  0.0)
    last_lap = S.get('lastLapTime', 0.0)

    def fmt_time(t):
        if t <= 0.0:
            return "--:--.---"
        m = int(t) // 60
        s = t - m * 60
        return f"{m}:{s:06.3f}"

    try:
        data = {
            "speedX":          S.get("speedX",   0.0),
            "speedY":          S.get("speedY",   0.0),
            "trackPos":        S.get("trackPos", 0.0),
            "angle":           S.get("angle",    0.0),
            "distFromStart":   S.get("distFromStart", 0.0),
            "distRaced":       S.get("distRaced", 0.0),
            "rpm":             S.get("rpm",  0.0),
            "gear":            R.get("gear", 1),
            "accel":           R.get("accel", 0.0),
            "brake":           R.get("brake", 0.0),
            "steer":           R.get("steer", 0.0),
            "damage":          S.get("damage", 0.0),
            "fuel":            round(S.get("fuel", 0.0), 2),
            "wheelSpinVel":    S.get("wheelSpinVel", [0, 0, 0, 0]),
            "wheelSkid":       skid,
            "tyreWearPct":     tyre_wear,
            "curLapTime":      round(cur_lap,  3),
            "lastLapTime":     round(last_lap, 3),
            "curLapTime_fmt":  fmt_time(cur_lap),
            "lastLapTime_fmt": fmt_time(last_lap),
            "sessionTime":     round(time.time() - _session_start_time, 1),
            "track":           S.get("track", []),
            "timestamp":       time.time(),
        }
        with open(TELEMETRY_FILE, "w") as f:
            _json.dump(data, f)
    except Exception:
        pass


def mode_drive():
    global _tyre_skid_accum, _session_start_time
    _tyre_skid_accum    = [0.0, 0.0, 0.0, 0.0]
    _session_start_time = None

    print(f"{TELEMETRY_FILE}")
    result = load_best_params()
    if result:
        params, fitness, _ = result
        print(f"Loaded params  fitness={fitness:.1f}")
    else:
        params = DriveParams()
        print("No saved params — using defaults.")

    try:
        import snakeoil3_gym as snakeoil3
    except ImportError:
        import snakeoil3_jm2 as snakeoil3

    C = snakeoil3.Client(p=3001)
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        drive(C, params)
        _write_telemetry(C.S.d, C.R.d)
        C.respond_to_server()
    C.shutdown()


def mode_optimise(continuous: bool = False):
    from gym_torcs import TorcsEnv

    result = load_best_params()
    if result:

        best_params, best_fitness, history = result
        SANE_FITNESS_CAP = 10_000_000.0

        if best_fitness > SANE_FITNESS_CAP:
            print(f"WARNING: saved fitness {best_fitness:.1f} looks corrupt. Resetting to 0.")
            best_fitness = 0.0
        print(f"Resuming from {RESULTS_FILE}  fitness={best_fitness:.1f}")
    else:
        best_params  = DriveParams()
        best_fitness = -np.inf
        history      = []
        print("Starting fresh.")

    sigma = MUTATION_SIGMA
    np.random.seed(42)
    env   = TorcsEnv(vision=False, throttle=False)

    try:
        if continuous:

            start_gen = history[-1]["generation"] if history else 0
            run_continuous_optimisation(
                env          = env,
                seed_params  = best_params,
                seed_fitness = best_fitness,
                history      = history,
                start_gen    = start_gen,
                sigma        = sigma,
            )


        else:

            round_number = (history[-1].get("round", 0) if history else 0) + 1
            best_params, best_fitness, history, sigma = run_optimisation_round(
                env          = env,
                seed_params  = best_params,
                seed_fitness = best_fitness,
                history      = history,
                round_number = round_number,
                sigma        = sigma,
            )
            _print_winner(best_params, best_fitness)


    except KeyboardInterrupt:
        
        print("\nInterrupted — checking saved results")
        saved = load_best_params()
        
        if saved and saved[1] >= best_fitness:
            print(f"  [keeping saved fitness={saved[1]:.1f}, in-memory was {best_fitness:.1f}]")
            _print_winner(saved[0], saved[1])
        
        else:
            save_results(best_params, best_fitness, history)
            print(f"  [saved to {RESULTS_FILE}]")
            _print_winner(best_params, best_fitness)
    
    finally:
        env.end()


if __name__ == "__main__":
    if _MODE == "drive":
        mode_drive()
    elif _MODE == "optimise":
        #optimise one round of NUM_GENERATIONS generations then exit
        mode_optimise(continuous=False)
    else:
        # continuous
        mode_optimise(continuous=True)
