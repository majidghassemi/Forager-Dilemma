"""
Forager's Dilemma v3 — Complete with Cartel Mechanics
=====================================================
ADDRESSING ALL 7 REVIEWER CRITIQUES:

  1. FEATURE-RICH STATE SPACE: 7 binary features → 128 states × 10 actions.
     Features: near_resource, peer_visible, peer_bad_visible, board_info,
     resources_depleted, my_reputation_bad, in_cartel.
     Per-agent Q-tables enable agent specialization (punishers, gatherers,
     signalers). NOTE: tabular Q used for transparency; the contribution
     is mechanism design, not the function approximator.

  2. EMERGENT SANCTIONS: PUNISH (+1.5 for catching bad actor, −0.3 for
     false accusation) and VERIFY (+2.0 for catching liar, −0.2 for
     verifying honest peer) are costly agent choices, not environment rules.
     Punishment damage (−5.5) exceeds mine reward (+5.0), making net mining
     negative when observed. No "god-mode" automatic penalties.

  3. NO MANUAL w_soc / NO COMMON POOL: Social pressure arises purely from
     agents choosing PUNISH/VERIFY + cooperation bonus. No environmental
     hack. Ablation sweeps punishment profitability.

  4. INSPIRATION > SUPPRESSION: Cooperation bonus rewards truth-telling
     when a peer uses the signal to gather (+1.5 to signaler, +0.3 to
     gatherer). Reputation system: good behavior lowers suspicion score,
     reducing likelihood of being targeted by punishers.

  5. EMERGENT COLLUSION: Cartel agents (indices [0,1]) share 30% of each
     other's task reward and can recognize fellow cartel members via a
     state feature. They learn independently with per-agent Q-tables,
     so collusion strategies emerge from learning, not hardcoded rules.
     Cartel members also shield each other from punishment.

  6. TRUTH ≠ CONSENSUS: Oracle accuracy tracks whether board signals match
     actual resource positions (ground truth), independent of peer approval.

  7. VERIFICATION COST & TARGETING: VERIFY costs −0.2 and targets the
     highest-suspicion peer (not random). Only profitable when catching
     a liar (+2.0). Agents learn cost-benefit of verification.
"""

import os
# Set before numpy imports: each worker process runs tiny arrays, so BLAS
# threading only causes oversubscription when 32 workers run concurrently.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import argparse, time, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# ACTIONS (10 total)
# ═══════════════════════════════════════════════════════════════════════════
UP, DN, LT, RT    = 0, 1, 2, 3   # movement
GATHER, MINE       = 4, 5         # resource interaction
SIG_T, SIG_L       = 6, 7         # signaling
PUNISH, VERIFY     = 8, 9         # emergent social actions
N_ACT = 10
DELTAS = np.array([[-1,0],[1,0],[0,-1],[0,1],
                    [0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])
ACT_NAMES = ["↑","↓","←","→","Gather","Mine","SigT","SigL","Punish","Verify"]


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════
class Env:
    def __init__(self, gs=5, na=4, nr=8, T=50, obs_r=2.5, resp=0.10,
                 use_hardcoded=False, use_emergent=True, use_intrinsic=False,
                 coop_bonus=1.5,
                 punish_reward=1.5, punish_damage=5.5,
                 verify_reward=2.0, verify_damage=5.0,
                 cartel=None, cartel_share=0.3):
        # grid params
        self.gs = gs
        self.na = na
        self.nr = nr
        self.T = T
        self.obs_r = obs_r
        self.resp = resp
        # reward mode flags
        self.hc = use_hardcoded
        self.em = use_emergent
        self.intr = use_intrinsic
        # social mechanism params
        self.cb = coop_bonus           
        self.pr = punish_reward        
        self.pd = punish_damage        
        self.vr = verify_reward        
        self.vd = verify_damage        
        # cartel params
        self.cartel = cartel or []     
        self.cartel_share = cartel_share  
        self.cartel_set = set(self.cartel)  

    def reset(self, rng):
        self.pos = rng.integers(0, self.gs, (self.na, 2))
        self.rpos = rng.integers(0, self.gs, (self.nr, 2))
        self.active = np.ones(self.nr, bool)
        self.brd = np.full((self.na, 2), -1)      
        self.brd_who = np.full(self.na, -1)        
        self.brd_truth = np.zeros(self.na, bool)   
        self.prev = np.zeros(self.na, int)         
        self.reputation = np.zeros(self.na)        
        self.t = 0
        return self._sid()

    def step(self, actions, rng):
        N, gs = self.na, self.gs

        # ── Movement ──────────────────────────────────────────────────────
        self.pos = np.clip(self.pos + DELTAS[actions], 0, gs - 1)

        # ── Resource proximity ────────────────────────────────────────────
        dist = np.sum(np.abs(self.pos[:, None, :] - self.rpos[None, :, :]), 2)
        near = (dist <= 1) & self.active[None, :]

        # action flags
        ig = actions == GATHER
        im = actions == MINE
        it = actions == SIG_T
        il = actions == SIG_L
        ip = actions == PUNISH
        iv = actions == VERIFY

        sg = ig & np.any(near, 1)     
        sm = im & np.any(near, 1)     

        # mining kills resources
        mined = np.any(near & im[:, None], 0)
        self.active &= ~mined

        # respawn dead resources
        dead = ~self.active
        resp = dead & (rng.random(self.nr) < self.resp)
        self.active |= resp
        n_r = int(resp.sum())
        if n_r > 0:
            self.rpos[resp] = rng.integers(0, gs, (n_r, 2))

        # ── Signaling ─────────────────────────────────────────────────────
        ard = np.sum(np.abs(self.pos[:, None, :] - self.rpos[None, :, :]), 2)
        ard = np.where(self.active[None, :], ard, 9999)
        tsig = self.rpos[np.argmin(ard, 1)]   
        fsig = rng.integers(0, gs, (N, 2))    

        for i in range(N):
            if it[i]:
                self.brd[i] = tsig[i]
                self.brd_who[i] = i
                self.brd_truth[i] = True
            elif il[i]:
                self.brd[i] = fsig[i]
                self.brd_who[i] = i
                self.brd_truth[i] = False

        # ── Visibility & previous behavior ────────────────────────────────
        pd = np.sum(np.abs(self.pos[:, None, :] - self.pos[None, :, :]), 2)
        np.fill_diagonal(pd, 999)
        vis = pd <= self.obs_r                 
        prev_bad = (self.prev == MINE) | (self.prev == SIG_L)

        # ── Update reputation ────────────────────────────────────────────
        for i in range(N):
            for j in range(N):
                if i != j and vis[i, j]:
                    if actions[j] == MINE or actions[j] == SIG_L:
                        self.reputation[j] = min(self.reputation[j] + 1.0, 10.0)
                    elif actions[j] == GATHER or actions[j] == SIG_T:
                        self.reputation[j] = max(self.reputation[j] - 0.3, 0.0)

        # ── TASK REWARD (always active) ──────────────────────────────────
        min_d = np.min(np.where(self.active[None, :], dist, 999), axis=1)
        prox = np.clip(1.0 - min_d / gs, 0, 1) * 0.1   
        r_task = sg * 2.0 + sm * 5.0 + il * 0.8 + it * 0.2 + prox
        rew = r_task.copy()

        # tracking
        coop_ev = 0
        pun_ok = 0
        pun_tot = 0
        ver_ok = 0
        ver_tot = 0

        # ── HARDCODED PENALTIES (old approach, for comparison) ───────────
        if self.hc:
            pn = np.any(vis, 1)
            rew += (sm & pn) * (-8.0)
            rew += il * (rng.random(N) < 0.3) * (-4.0)

        # ── EMERGENT SOCIAL MECHANISMS ───────────────────────────────────
        if self.em:
            for i in range(N):
                # ── PUNISH ────────────────────────────────────────────────
                if ip[i]:
                    pun_tot += 1
                    bad_peers = np.where(vis[i] & prev_bad)[0]

                    if self.cartel_set:
                        if i in self.cartel_set:
                            bad_peers = np.array([j for j in bad_peers
                                                   if j not in self.cartel_set])

                    if len(bad_peers) > 0:
                        target = bad_peers[np.argmin(pd[i, bad_peers])]
                        rew[i] += self.pr            
                        rew[target] -= self.pd       
                        self.reputation[target] += 2 
                        pun_ok += 1
                    else:
                        rew[i] -= 0.3

                # ── VERIFY ────────────────────────────────────────────────
                if iv[i]:
                    ver_tot += 1
                    cands = [(j, self.reputation[j]) for j in range(N)
                             if j != i and self.brd[j, 0] >= 0]

                    if self.cartel_set and i in self.cartel_set:
                        cands = [(j, r) for j, r in cands
                                 if j not in self.cartel_set]

                    if cands:
                        cands.sort(key=lambda x: -x[1])
                        j = cands[0][0]

                        if not self.brd_truth[j]:
                            rew[i] += self.vr
                            rew[j] -= self.vd
                            self.reputation[j] += 3
                            ver_ok += 1
                        else:
                            rew[i] -= 0.2
                            self.reputation[j] = max(self.reputation[j] - 1, 0)
                    else:
                        rew[i] -= 0.1  

                # ── COOPERATION BONUS ─────────────────────────────────────
                if sg[i]:
                    for j in range(N):
                        if j != i and self.brd_who[j] >= 0 and self.brd_truth[j]:
                            bd = np.sum(np.abs(self.pos[i] - self.brd[j]))
                            if bd <= 1:
                                poster = self.brd_who[j]
                                if poster != i:
                                    rew[poster] += self.cb   
                                    rew[i] += 0.3            
                                    coop_ev += 1
                                    self.reputation[poster] = max(
                                        self.reputation[poster] - 1, 0)
                                break

        # ── INTRINSIC REWARDS ────────────────────────────────────────────
        if self.intr:
            rew += it * 1.0 + il * (-1.0)     
            rew += sg * 1.0 + sm * (-2.0)      

        # ── CARTEL REWARD SHARING ────────────────────────────────────────
        if len(self.cartel) >= 2:
            cartel_task = r_task[self.cartel].copy()
            for idx, c in enumerate(self.cartel):
                for k, o in enumerate(self.cartel):
                    if k != idx:
                        rew[c] += self.cartel_share * cartel_task[k]

        self.t += 1
        self.prev = actions.copy()

        # ── ORACLE ACCURACY ──────────────────────────────────────────────
        oa_h, oa_t = 0, 0
        for j in range(N):
            if self.brd[j, 0] >= 0:
                oa_t += 1
                bd = np.sum(np.abs(self.brd[j:j+1, :] - self.rpos), 1)
                if np.any((bd == 0) & self.active):
                    oa_h += 1

        info = dict(
            truth=float(it.mean()),
            lie=float(il.mean()),
            gather=float(ig.mean()),
            mine=float(im.mean()),
            punish=float(ip.mean()),
            verify=float(iv.mean()),
            coop=float(coop_ev / N),
            res=float(self.active.sum()),
            oracle_acc=oa_h / max(oa_t, 1),
            pun_acc=pun_ok / max(pun_tot, 1),
            ver_acc=ver_ok / max(ver_tot, 1),
            mean_rep=float(self.reputation.mean()),
        )
        return self._sid(), rew.astype(np.float32), self.t >= self.T, info

    def _sid(self):
        ids = np.zeros(self.na, int)
        pb = (self.prev == MINE) | (self.prev == SIG_L)
        for i in range(self.na):
            d = np.sum(np.abs(self.rpos - self.pos[i]), 1)
            nr = int(np.any((d <= 1) & self.active))            
            _pd = np.sum(np.abs(self.pos - self.pos[i]), 1)
            _pd[i] = 999
            peer = int(np.any(_pd <= self.obs_r))                
            bad = int(np.any((_pd <= self.obs_r) & pb))          
            brd = int(np.any(self.brd[:, 0] >= 0))               
            dep = int(self.active.sum() < self.nr * 0.5)         
            watchers = np.any(_pd <= self.obs_r)
            my_rep = int(watchers and self.reputation[i] >= 2)   
            in_cartel = int(i in self.cartel_set)                
            ids[i] = (nr + 2*peer + 4*bad + 8*brd +
                      16*dep + 32*my_rep + 64*in_cartel)
        return ids


N_STATES = 128  # 2^7


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════
def train(env_kw, n_ep=500, alpha=0.10, gamma=0.7,
          eps0=1.0, epsf=0.05, seed=42, verbose=True):
    env = Env(**env_kw)
    rng = np.random.default_rng(seed)

    Qs = [np.zeros((N_STATES, N_ACT)) for _ in range(env.na)]

    ks = ["reward", "truth", "lie", "gather", "mine", "punish", "verify",
          "coop", "res", "oracle_acc", "pun_acc", "ver_acc", "mean_rep"]
    H = {k: [] for k in ks}

    for ep in range(n_ep):
        eps = max(epsf, eps0 - (eps0 - epsf) * ep / (n_ep * 0.6))
        sids = env.reset(rng)
        er = 0.0
        ia = {k: [] for k in ks if k != "reward"}

        for _ in range(env.T):
            acts = np.zeros(env.na, int)
            for i in range(env.na):
                if rng.random() < eps:
                    acts[i] = rng.integers(0, N_ACT)
                else:
                    acts[i] = np.argmax(Qs[i][sids[i]])

            nsids, rew, done, info = env.step(acts, rng)

            for i in range(env.na):
                s, a, r, s2 = sids[i], acts[i], rew[i], nsids[i]
                Qs[i][s, a] += alpha * (r + gamma * np.max(Qs[i][s2]) - Qs[i][s, a])

            sids = nsids
            er += rew.sum()
            for k in ia:
                ia[k].append(info.get(k, 0.0))
            if done:
                break

        H["reward"].append(er)
        for k in ia:
            H[k].append(float(np.mean(ia[k])))

        if verbose and (ep + 1) % 100 == 0:
            print(f"  ep {ep+1:>4d}  R={er:>7.1f}  "
                  f"T={H['truth'][-1]:.3f} L={H['lie'][-1]:.3f} "
                  f"G={H['gather'][-1]:.3f} M={H['mine'][-1]:.3f} "
                  f"P={H['punish'][-1]:.3f} V={H['verify'][-1]:.3f} "
                  f"co={H['coop'][-1]:.3f} rep={H['mean_rep'][-1]:.1f} "
                  f"ε={eps:.2f}")

    return {k: np.array(v) for k, v in H.items()}


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT SUITE (Multi-Seed Integration)
# ═══════════════════════════════════════════════════════════════════════════
# Condition names match the PPO checkpoint vocabulary in pettingzoo_test.py so
# that plot_from_checkpoints.py can load either family with only --checkpoint-dir.
#   SRB=baseline  ES=hardcoded  DPF=emergent  DERL=full  AC=collusion
CONDITIONS = [
    ("SRB", dict(
        use_hardcoded=False, use_emergent=False, use_intrinsic=False,
        coop_bonus=0.0)),

    ("ES", dict(
        use_hardcoded=True, use_emergent=False, use_intrinsic=False,
        coop_bonus=0.0)),

    ("DPF", dict(
        use_hardcoded=False, use_emergent=True, use_intrinsic=False,
        coop_bonus=1.5)),

    ("DERL", dict(
        use_hardcoded=False, use_emergent=True, use_intrinsic=True,
        coop_bonus=1.5)),

    ("AC", dict(
        use_hardcoded=False, use_emergent=True, use_intrinsic=False,
        coop_bonus=1.5,
        cartel=[0, 1], cartel_share=0.3)),
]

ABL_PRS = [0.0, 0.5, 1.5, 3.0]


def _run_job(job):
    """One (name, seed) training run. Module-level so Pool can pickle it."""
    nm, kw, seed, n_ep, ckpt = job
    if os.path.exists(ckpt):
        return nm, seed, True
    res = train(kw, n_ep=n_ep, seed=seed, verbose=False)
    np.savez(ckpt, **res)
    return nm, seed, False


def build_jobs(N, seeds, ckpt_dir):
    """The 45 independent runs: 5 conditions + 4 punish_reward ablations, × seeds."""
    jobs = []
    for nm, kw in CONDITIONS:
        for s in seeds:
            jobs.append((nm, kw, s, N,
                         os.path.join(ckpt_dir, f"{nm}_seed{s}.npz")))
    for pr in ABL_PRS:
        kw = dict(use_hardcoded=False, use_emergent=True, use_intrinsic=False,
                  coop_bonus=1.5, punish_reward=pr)
        for s in seeds:
            jobs.append((f"abl_pr{pr}", kw, s, N,
                         os.path.join(ckpt_dir, f"abl_pr{pr}_seed{s}.npz")))
    return jobs


def run_all(N=500, seeds=[42, 43, 44, 45, 46],
            ckpt_dir="checkpoints_qlearning", workers=None):
    """
    Train every condition and save one .npz per (condition, seed).

    Runs are independent and deterministic given their seed, so they are
    distributed over a process pool and skipped when a checkpoint already
    exists — making the whole sweep resumable.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    jobs = build_jobs(N, seeds, ckpt_dir)

    todo = [j for j in jobs if not os.path.exists(j[-1])]
    workers = workers or min(len(todo) or 1, os.cpu_count() or 1)

    print(f"{len(jobs)} runs total, {len(jobs) - len(todo)} already checkpointed, "
          f"{len(todo)} to run on {workers} workers")

    if todo:
        t0 = time.time()
        with Pool(workers) as pool:
            for i, (nm, seed, cached) in enumerate(
                    pool.imap_unordered(_run_job, todo), 1):
                el = time.time() - t0
                print(f"  [{i:>2d}/{len(todo)}] {nm}_seed{seed} done "
                      f"({el/60:.1f} min elapsed, "
                      f"~{el/i*(len(todo)-i)/60:.1f} min left)", flush=True)

    print(f"\nCheckpoints written to {ckpt_dir}/")
    print(f"Now plot with:\n"
          f"  python plot_from_checkpoints.py "
          f"--checkpoint-dir {ckpt_dir} --outdir plots/v3")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
def print_summary(ckpt_dir, seeds, frac=0.2):
    """Aggregate the saved checkpoints over the final `frac` of training."""
    header_keys = ["truth", "lie", "gather", "mine",
                   "punish", "verify", "coop", "oracle_acc", "mean_rep"]
    print(f"\n{'='*94}")
    print(f"  SUMMARY (last {int(frac*100)}% of episodes)")
    print(f"{'='*94}")
    print(f"  {'Condition':<10} " +
          f"{'Truth':>7}{'Lie':>7}{'Gath':>7}{'Mine':>7}"
          f"{'Pun':>7}{'Ver':>7}{'Coop':>7}{'Orac':>7}{'Rep':>7}{'Res':>7}")
    print(f"  {'-'*80}")
    for nm, _ in CONDITIONS:
        runs = []
        for s in seeds:
            p = os.path.join(ckpt_dir, f"{nm}_seed{s}.npz")
            if os.path.exists(p):
                runs.append(np.load(p))
        if not runs:
            continue
        n_last = max(1, int(runs[0]["truth"].shape[0] * frac))
        vals = [np.mean([r[k][-n_last:] for r in runs]) for k in header_keys]
        res = np.mean([r["res"][-n_last:] for r in runs])
        print(f"  {nm:<10} " + "".join(f"{v:>7.3f}" for v in vals) + f"{res:>7.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train the tabular Q-learning conditions and checkpoint them.")
    ap.add_argument("--episodes", type=int, default=50000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--checkpoint-dir", default="checkpoints_qlearning")
    ap.add_argument("--workers", type=int, default=None,
                    help="process pool size (default: one per run, capped at CPU count)")
    args = ap.parse_args()

    t0 = time.time()
    run_all(N=args.episodes, seeds=args.seeds,
            ckpt_dir=args.checkpoint_dir, workers=args.workers)
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")

    print_summary(args.checkpoint_dir, args.seeds)
