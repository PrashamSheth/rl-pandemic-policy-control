"""
Agent-based pandemic simulation with mobility-control policies.

The simulation, the PI and PID controllers, the state vector and the DQN
network are the same as in pandemic_policy_rl.ipynb (DQN+PI v3), moved into
one module so the saved model can be evaluated without retraining.
"""

import numpy as np
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn

# =============================================================================
# 1. GLOBAL PARAMETERS
# =============================================================================
NUM_AGENTS     = 2500
GRID_SIZE      = 1.0
DAYS           = 200
CAPACITY_LIMIT = 500

R_AVG         = 1.0 / np.sqrt(NUM_AGENTS)
INF_RADIUS    = R_AVG / 5.0
INF_RADIUS_SQ = INF_RADIUS ** 2

L_LOCKDOWN = 0.016
L_OPEN     = 2.0 * R_AVG
L_RANGE    = L_OPEN - L_LOCKDOWN

PI_K          = 6
PI_LOOK_AHEAD = 3
PID_KD        = 0.8

BLEND_WEIGHTS = [-0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5, 0.8, 1.0]
NUM_ACTIONS   = len(BLEND_WEIGHTS)
STATE_DIM     = 10


# =============================================================================
# 2. SIMULATION PRIMITIVES
# =============================================================================

def infect_grid(positions, status, inf_radius_sq, grid_size=20):
    """Grid-bucketed infection check: only neighbouring cells are compared."""
    G = grid_size
    cells     = [[] for _ in range(G * G)]
    inf_cells = [[] for _ in range(G * G)]
    for i in range(len(positions)):
        gx  = min(G-1, int(positions[i, 0] * G))
        gy  = min(G-1, int(positions[i, 1] * G))
        idx = gy * G + gx
        (inf_cells if status[i] == 1 else cells)[idx].append(i)
    new_infected = []
    for cy in range(G):
        for cx in range(G):
            sus = cells[cy * G + cx]
            if not sus: continue
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = cy+dy, cx+dx
                    if not (0 <= ny < G and 0 <= nx < G): continue
                    infs = inf_cells[ny * G + nx]
                    for si in sus:
                        if status[si] == 1: continue
                        for ii in infs:
                            d2 = ((positions[si,0]-positions[ii,0])**2
                                + (positions[si,1]-positions[ii,1])**2)
                            if d2 <= inf_radius_sq:
                                new_infected.append(si); break
    for si in new_infected: status[si] = 1


def move_agents(positions, l):
    """Random walk with step size l and reflective boundaries."""
    a = np.random.uniform(0, 2*np.pi, len(positions))
    positions[:, 0] += l * np.cos(a)
    positions[:, 1] += l * np.sin(a)
    positions[:] = np.abs(positions)
    positions[:] = np.where(positions > GRID_SIZE, 2*GRID_SIZE - positions, positions)


def seed_population(fixed_seed=None):
    """Uniform random positions, one infected agent nearest the centre."""
    rng = np.random.RandomState(fixed_seed) if fixed_seed is not None else np.random
    positions = rng.rand(NUM_AGENTS, 2)
    status    = np.zeros(NUM_AGENTS, dtype=np.uint8)
    status[np.argmin(cdist(positions, [[0.5, 0.5]]))] = 1
    return positions, status


# =============================================================================
# 3. PI CONTROLLER
# =============================================================================

def pi_step(current_inf, recent_inf, prev_l, integral,
            k=PI_K, look_ahead=PI_LOOK_AHEAD):
    cap_ratio = current_inf / CAPACITY_LIMIT
    trend = 0.0
    if len(recent_inf) >= 2:
        trend = (recent_inf[-1] - recent_inf[0]) / (len(recent_inf)-1) / CAPACITY_LIMIT
    if cap_ratio > 1.0:
        integral = min(integral + (cap_ratio - 1.0) * 0.05, 1.0)
    else:
        integral = max(0.0, integral - 0.01)
    predictive = cap_ratio + look_ahead * max(0.0, trend)
    if predictive < 0.3:
        target_l = L_OPEN
    elif predictive < 0.8:
        t = (predictive - 0.3) / 0.5
        target_l = L_OPEN * np.exp(-k * t)
    else:
        target_l = L_OPEN * np.exp(-k) * np.exp(-k * integral)
    hyst = 0.6 if prev_l < L_OPEN * 0.5 else 1.0
    if cap_ratio > hyst * 0.8:
        target_l = min(target_l, prev_l)
    max_delta = 0.15 * L_OPEN
    l = float(np.clip(target_l, prev_l - max_delta, prev_l + max_delta))
    l = float(np.clip(l, L_LOCKDOWN * 0.5, L_OPEN))
    return l, integral


# =============================================================================
# 4. PID CONTROLLER (PI plus a derivative correction)
# =============================================================================

def pid_step(current_inf, recent_inf, prev_l, integral, prev_error,
             k=PI_K, look_ahead=PI_LOOK_AHEAD, kd=PID_KD):
    pi_l, integral = pi_step(current_inf, recent_inf, prev_l, integral, k, look_ahead)
    setpoint = 0.7
    error    = (current_inf / CAPACITY_LIMIT) - setpoint
    d_error  = error - prev_error
    deriv_correction = kd * d_error * L_RANGE
    pid_l = float(np.clip(pi_l - deriv_correction, L_LOCKDOWN * 0.5, L_OPEN))
    return pid_l, integral, error


# =============================================================================
# 5. STATE VECTOR (10 features)
# =============================================================================

def build_state_vector(current_inf, inf_history, pi_l, pid_l, prev_l, day):
    h = list(inf_history)
    cap_ratio = current_inf / CAPACITY_LIMIT
    f0 = float(np.sqrt(np.clip(cap_ratio / 2.0, 0.0, 1.0)))
    accel = (h[-1] - 2*h[-2] + h[-3]) / CAPACITY_LIMIT if len(h) >= 3 else 0.0
    f1 = float(np.clip(accel / 0.10, -1.0, 1.0))
    if len(h) >= 7:
        t7 = (h[-1] - h[-7]) / (6.0 * CAPACITY_LIMIT)
    elif len(h) >= 2:
        t7 = (h[-1] - h[0]) / (max(len(h)-1, 1) * CAPACITY_LIMIT)
    else:
        t7 = 0.0
    f2 = float(np.clip(t7 / 0.10, -1.0, 1.0))
    f3 = float((pi_l  - L_LOCKDOWN) / L_RANGE)
    f4 = float((pid_l - L_LOCKDOWN) / L_RANGE)
    f5 = float(np.clip((pid_l - pi_l) / L_RANGE, -1.0, 1.0))
    f6 = float((prev_l - L_LOCKDOWN) / L_RANGE)
    f7 = float(1.0 - (pi_l - L_LOCKDOWN) / L_RANGE)
    f8 = float(day / DAYS)
    f9 = float(1.0 - current_inf / NUM_AGENTS)
    return np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9], dtype=np.float32)


# =============================================================================
# 6. DUELING DQN
# =============================================================================

class DQN(nn.Module):
    def __init__(self, state_dim=STATE_DIM, num_actions=NUM_ACTIONS, hidden=96):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(nn.Linear(hidden, 48), nn.ReLU(), nn.Linear(48, 1))
        self.adv_stream   = nn.Sequential(nn.Linear(hidden, 48), nn.ReLU(), nn.Linear(48, num_actions))

    def forward(self, x):
        h = self.shared(x)
        v = self.value_stream(h)
        a = self.adv_stream(h)
        return v + a - a.mean(dim=-1, keepdim=True)


def load_trained_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = DQN(state_dim=checkpoint["state_dim"], num_actions=checkpoint["num_actions"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# =============================================================================
# 7. ONE EPISODE UNDER A GIVEN POLICY
# =============================================================================

POLICIES = ["Always_Open", "Manual_Reactive", "Always_Lockdown", "Dynamic_PI", "PID", "DQN_PI"]


def run_policy(policy_name, policy_net=None):
    """Run one 200-day episode. Returns (infected per day, step size per day)."""
    positions, status = seed_population()
    prev_l = L_OPEN; integral = 0.0; recent = []; prev_err = 0.0
    inf_history = [1] * 8
    history_inf, history_mob = [], []

    for day in range(DAYS):
        current_inf = int(np.sum(status))
        inf_history.append(current_inf)
        if len(inf_history) > 8: inf_history.pop(0)
        recent.append(current_inf)
        if len(recent) > PI_LOOK_AHEAD + 1: recent.pop(0)

        if policy_name == "DQN_PI":
            pi_l, _ = pi_step(current_inf, recent, prev_l, 0.0)
            pid_l, integral, prev_err = pid_step(current_inf, recent, prev_l, integral, prev_err)
            state = build_state_vector(current_inf, inf_history, pi_l, pid_l, prev_l, day)
            with torch.no_grad():
                a_idx = int(policy_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item())
            w = BLEND_WEIGHTS[a_idx]
            l = float(np.clip(pi_l + w*(L_OPEN - pi_l), L_LOCKDOWN*0.5, L_OPEN))
        elif policy_name == "Always_Open":
            l = L_OPEN
        elif policy_name == "Always_Lockdown":
            l = L_LOCKDOWN
        elif policy_name == "Manual_Reactive":
            l = L_LOCKDOWN if current_inf > 0.8 * CAPACITY_LIMIT else L_OPEN
        elif policy_name == "Dynamic_PI":
            l, integral = pi_step(current_inf, recent, prev_l, integral)
        elif policy_name == "PID":
            l, integral, prev_err = pid_step(current_inf, recent, prev_l, integral, prev_err)
        else:
            raise ValueError(policy_name)

        move_agents(positions, l)
        infect_grid(positions, status, INF_RADIUS_SQ)
        history_inf.append(current_inf)
        history_mob.append(l)
        prev_l = l

    return np.array(history_inf), np.array(history_mob)


def episode_metrics(history, mobility):
    """Peak infections, total mobility, days over capacity and efficiency.

    Efficiency = total mobility / (total infected-days / 1000). The original
    ensemble version divided the per-day mobility array instead of its sum,
    which is why the notebook prints 0.0001 for the PI, PID and DQN rows.
    """
    h = np.asarray(history); m = np.asarray(mobility)
    total_mob = float(np.sum(m))
    return {"Peak": int(np.max(h)),
            "Mobility": total_mob,
            "Days Over Cap": int(np.sum(h > CAPACITY_LIMIT)),
            "Efficiency": total_mob / (float(np.sum(h)) / 1000 + 1e-9)}
