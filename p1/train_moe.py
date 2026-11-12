#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_moe.py

MoE training script (modified from the original baseline+fuzzy.py). Four ways of controlling the routing hyperparameters share the same model, data and training code:

    --controller fixed     τ, λ, σ fixed throughout (default 0.5, 1.5, 0.1)
    --controller schedule  τ, λ, σ change linearly per epoch after warmup ends (not fuzzy, does not read the training state)
    --controller fuzzy     the fuzzy controller adjusts τ, λ, σ every epoch from validation signals
    --controller auxfree   auxiliary-loss-free load balancing as in DeepSeek-V3: affinity sigmoid(s/τ), top-k selected by affinity + bias,
                           weights use affinity only; the bias is adjusted every step from expert load. τ, σ fixed, default λ=0 (pilot stage 6 uses σ=0)

Main differences from the original script:
- Reads CSV file lists; linear warmup + cosine annealing; early stopping removed; resume from checkpoint removed
- Validation now computes expert outputs per batch (mathematically equivalent to the original per-sample loop)
- Training, validation and inference all use the current τ (the original script fixed τ=1 at inference)
- Router scoring changed to cosine similarity (following the Tutel CosineTopKGate used by GMoE, projection dimension 16, choice in analysis/TODO.md "Cosine router projection dimension"),
  replacing the original conv3 + LayerNorm. Scores lie in [-1, 1] and do not change with the scale of learnable parameters; τ is the only scale;
  τ is not learnable and is given by the controller. In non-auxfree modes τ does not change the top-k selection, only the weights among the selected experts
- Trends T_fast, T_slow are computed by Eq. (12) of the paper: T = mean_{i=1..W}(a_t - a_{t-i})
- Controller inputs use the validation accuracy A_π weighted by the training-subset target distribution π, and the weighted expert usage that is combined first and then used to compute balance
- The controller starts adjusting only after the learning-rate warmup ends; the aux loss warmup is converted as a fraction of the total number of epochs
- Checkpoint selected by validation A_π; each epoch logs the controller inputs, memberships, rule activations and outputs
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn

from common import (ENV_LOCK_DEFAULT, CSVImageDataset, CSVLogger, TrainDiagnostics, build_scheduler, build_transforms, check_env,
                    classification_metrics, default_num_workers, load_manifest,
                    make_loader, prepare_out_dir, save_json, set_seed, weighted_expert_usage)

NUM_EXPERTS = 5
TOP_K = 2
SPLIT_BLOCK = 6  # blocks[:6] are the shared feature extractor, blocks[6:] are the experts
ROUTER_PROJ_DIM = 16  # projection dimension of the cosine router. Expert directions can be pairwise equidistant only when d >= NUM_EXPERTS - 1, but this is not necessary for routing to work;
                      # the X-MoE rule of thumb is d_e = N/2 (16 for its 32 experts). Discussion of the choice in analysis/TODO.md "Cosine router projection dimension"

# Legacy controller parameters (rules of Table I in the submitted paper and membership functions of the original baseline+fuzzy.py, τ-related terms scaled by 0.5/0.9), only for reproducing old runs
# (--fuzzy-config legacy). The design from stage 8 on is in fuzzy_v2.json (default), with rules also read from the JSON; see analysis/Pilot_Study.md stage 8.
_TAU_SCALE = 0.5 / 0.9
FUZZY_CONFIG_DEFAULT = Path(__file__).resolve().parent / "fuzzy_v2.json"
FUZZY_LEGACY = {
    "inputs": {
        "B": {"range": [70, 100, 1], "terms": {
            "Unstable": ["trap", [70, 70, 78, 84]],
            "Normal": ["tri", [80, 88, 94]],
            "Perfect": ["trap", [91, 95, 100, 100]]}},
        "T_fast": {"range": [-0.02, 0.03, 0.001], "terms": {
            "Degrading": ["trap", [-0.02, -0.02, -0.008, 0.0]],
            "Stable": ["tri", [-0.005, 0.003, 0.012]],
            "Improving": ["trap", [0.008, 0.015, 0.03, 0.03]]}},
        "T_slow": {"range": [-0.02, 0.05, 0.001], "terms": {
            "Degrading": ["trap", [-0.02, -0.02, -0.005, 0.002]],
            "Stable": ["tri", [-0.003, 0.003, 0.012]],
            "Improving": ["trap", [0.005, 0.018, 0.05, 0.05]]}},
    },
    "outputs": {
        "delta_tau": {"range": [v * _TAU_SCALE for v in (-0.04, 0.04, 0.001)], "terms": {
            "Decrease": ["tri", [v * _TAU_SCALE for v in (-0.04, -0.025, 0.0)]],
            "Keep": ["tri", [v * _TAU_SCALE for v in (-0.01, 0.0, 0.01)]],
            "Increase": ["tri", [v * _TAU_SCALE for v in (0.0, 0.025, 0.04)]]}},
        "lambda_out": {"range": [0.5, 2.5, 0.01], "terms": {
            "Low": ["trap", [0.5, 0.5, 0.8, 1.2]],
            "Medium": ["tri", [0.8, 1.2, 1.8]],
            "High": ["trap", [1.5, 2.0, 2.5, 2.5]]}},
        "noise": {"range": [0.0, 0.15, 0.001], "terms": {
            "Low": ["trap", [0.0, 0.0, 0.015, 0.04]],
            "Medium": ["tri", [0.025, 0.06, 0.10]],
            "High": ["trap", [0.08, 0.12, 0.15, 0.15]]}},
    },
    "window_fast": 3,
    "window_slow": 5,
    "beta": 0.8,
    "tau_min": 0.85 * _TAU_SCALE,
    "tau_max": 1.02 * _TAU_SCALE,
    "rules": [  # Table I of the paper; op = or takes the max over the conditions
        {"name": "R1", "op": "or", "if": {"B": "Unstable", "T_fast": "Degrading"}, "then": {"delta_tau": "Increase", "lambda_out": "High", "noise": "High"}},
        {"name": "R2", "if": {"B": "Normal", "T_fast": "Stable"}, "then": {"delta_tau": "Keep", "lambda_out": "Medium", "noise": "High"}},
        {"name": "R3", "if": {"B": "Perfect", "T_fast": "Improving"}, "then": {"delta_tau": "Decrease", "lambda_out": "Low", "noise": "Low"}},
        {"name": "R4", "if": {"T_slow": "Stable"}, "then": {"delta_tau": "Keep", "lambda_out": "Medium", "noise": "High"}},
        {"name": "R5", "if": {"B": "Normal", "T_fast": "Improving"}, "then": {"delta_tau": "Decrease", "lambda_out": "Medium", "noise": "High"}},
        {"name": "R6", "if": {"B": "Perfect", "T_fast": "Stable"}, "then": {"delta_tau": "Keep", "lambda_out": "Low", "noise": "High"}},
    ],
}


# ======================================================================
# model
# ======================================================================
class SharedFeatureExtractor(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.conv_stem = base.conv_stem
        self.bn1 = base.bn1
        self.act1 = getattr(base, "act1", nn.Identity())  # in timm>=0.6 bn1 already includes the activation, no act1
        self.blocks = nn.Sequential(*base.blocks[:SPLIT_BLOCK])

    def forward(self, x):
        return self.blocks(self.act1(self.bn1(self.conv_stem(x))))


class Router(nn.Module):
    def __init__(self, in_channels, num_experts):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.15)
        # cosine scoring head (Tutel CosineTopKGate): both the sample projection and the expert vectors are L2-normalized
        self.proj = nn.Linear(128, ROUTER_PROJ_DIM)
        self.expert_emb = nn.Parameter(torch.empty(ROUTER_PROJ_DIM, num_experts))
        nn.init.normal_(self.expert_emb, 0, 0.01)
        self.register_buffer("noise_scale", torch.tensor(0.1))

    def forward(self, x):
        """Return the cosine score of each expert (B, N_e), in [-1, 1]; noise is added during training (before dividing by τ, so that σ and τ are decoupled)."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        h = self.gap(x).flatten(1).float()
        with torch.autocast(x.device.type, enabled=False):  # as in Tutel, scores are computed in fp32
            q = F.normalize(self.proj(h), dim=-1)            # sample direction, (B, d)
            k = F.normalize(self.expert_emb, dim=0)          # expert direction, (d, N_e)
            logits = q @ k                                   # cosine similarity, (B, N_e)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_scale
        return logits


class CosineRouter(nn.Module):
    """R4: global average pooling of the shared features followed directly by the cosine scoring head, no conv, BN, ReLU, Dropout (same as Tutel CosineTopKGate, but τ is not learnable).
    The cosine head is the same as in Router, copied rather than shared, to keep the code and numerics of R0 unchanged."""

    def __init__(self, in_channels, num_experts):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_channels, ROUTER_PROJ_DIM)
        self.expert_emb = nn.Parameter(torch.empty(ROUTER_PROJ_DIM, num_experts))
        nn.init.normal_(self.expert_emb, 0, 0.01)
        self.register_buffer("noise_scale", torch.tensor(0.1))

    def forward(self, x):
        """Return the cosine score of each expert (B, N_e), in [-1, 1]; noise is added during training (before dividing by τ)."""
        h = self.gap(x).flatten(1).float()
        with torch.autocast(x.device.type, enabled=False):
            q = F.normalize(self.proj(h), dim=-1)
            k = F.normalize(self.expert_emb, dim=0)
            logits = q @ k
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_scale
        return logits


ROUTERS = {"R0": Router, "R4": CosineRouter}  # variant definitions in analysis/TODO.md; stage 5 compares only these two


class ExpertNetwork(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        self.blocks = nn.Sequential(*base.blocks[SPLIT_BLOCK:])
        self.conv_head = base.conv_head
        self.bn2 = base.bn2
        self.act2 = getattr(base, "act2", nn.Identity())  # in timm>=0.6 bn2 already includes the activation, no act2
        self.global_pool = base.global_pool
        self.classifier = nn.Linear(base.num_features, num_classes)

    def forward(self, x):
        x = self.act2(self.bn2(self.conv_head(self.blocks(x))))
        return self.classifier(self.global_pool(x))


class MoEModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, use_bias=False, tau=0.5, router="R0"):
        super().__init__()
        self.num_classes = num_classes
        self.use_bias = use_bias
        base = timm.create_model("efficientnet_b0", pretrained=pretrained)
        self.shared_extractor = SharedFeatureExtractor(base)
        self.shared_extractor.eval()  # in eval mode BN does not update the running statistics, so probing does not change the pretrained statistics
        with torch.no_grad():
            c = self.shared_extractor(torch.zeros(1, 3, 224, 224)).shape[1]
        self.shared_extractor.train()
        self.router = ROUTERS[router](c, NUM_EXPERTS)
        self.experts = nn.ModuleList([
            ExpertNetwork(timm.create_model("efficientnet_b0", pretrained=pretrained), num_classes)
            for _ in range(NUM_EXPERTS)
        ])
        self.register_buffer("tau", torch.tensor(float(tau)))
        # routing bias for auxfree mode; affects only the top-k selection, not the expert weighting coefficients
        self.register_buffer("route_bias", torch.zeros(NUM_EXPERTS))
        # Disturbance trial (disturb-pilot branch): external bias added to the top-k selection scores during training; does not affect the mixing weights and is not added during validation.
        # persistent=False: not in state_dict, checkpoint format unchanged. When disturb_on is False, forward is identical to the original code
        self.register_buffer("disturb_bias", torch.zeros(NUM_EXPERTS), persistent=False)
        self.disturb_on = False

    def forward(self, x):
        """Return (logits, router_probs, topk_idx, expert_logits). Training and validation use the same per-batch path.
        expert_logits (B, K, C) are the logits of each selected expert, column k corresponds to topk_idx[:, k]; logits is their weighted sum by w.
        In auxfree mode router_probs are the sigmoid affinities of DeepSeek-V3 Eq. (15) (independent per expert, not summing to 1); in the other modes they are softmax probabilities."""
        feats = self.shared_extractor(x)
        router_logits = self.router(feats)
        if self.use_bias:  # auxfree: select top-k by affinity + bias, weights use affinity only (DeepSeek-V3 Eqs. 13, 16)
            probs = torch.sigmoid(router_logits / self.tau)
            scores = probs + self.route_bias
        else:
            probs = F.softmax(router_logits / self.tau, dim=-1)
            scores = probs
        if self.training and self.disturb_on:
            scores = scores + self.disturb_bias
        topk_idx = torch.topk(scores, TOP_K, dim=-1).indices
        w = mix_weights(probs, topk_idx)
        mask = torch.zeros_like(probs).scatter(1, topk_idx, w)  # (B, N_e)
        out = torch.zeros(x.size(0), self.num_classes, device=x.device, dtype=torch.float32)
        expert_logits = torch.zeros(x.size(0), TOP_K, self.num_classes, device=x.device, dtype=torch.float32)
        for e in torch.unique(topk_idx).tolist():
            idx = torch.where((topk_idx == e).any(dim=1))[0]
            y = self.experts[e](feats[idx]).float()
            out = out.index_add(0, idx, mask[idx, e].unsqueeze(1) * y)
            expert_logits[idx, (topk_idx[idx] == e).int().argmax(dim=1)] = y
        return out, probs, topk_idx, expert_logits


def mix_weights(probs, topk_idx):
    """Normalized mixing weights w (B, K) of the selected experts; each row sums to 1."""
    w = probs.gather(1, topk_idx)
    return w / w.sum(dim=-1, keepdim=True)


def classification_loss(mode, out, expert_logits, w, y):
    """Classification loss (--cls-loss). mix: CE on the mixed logits (original implementation, cooperative: experts fill in each other's residuals).
    The other three are competitive (Jacobs et al. 1991): mixprob = -log Σ_k w_k p_k(y), negative log-likelihood of the mixture probability;
    expert = Σ_k w_k · CE_k, each expert is independently responsible for the label, router gradient ∂L/∂w_k = CE_k;
    oracle = CE of the selected expert with the smallest CE, plus the router's -log w_k for it (only the winning expert is updated)."""
    if mode == "mix":
        return F.cross_entropy(out, y)
    logp = F.log_softmax(expert_logits.float(), dim=-1)
    nll = -logp.gather(2, y[:, None, None].expand(-1, expert_logits.size(1), 1)).squeeze(2)  # (B, K) CE of each expert
    if mode == "mixprob":
        return -torch.logsumexp(torch.log(w) - nll, dim=1).mean()
    if mode == "expert":
        return (w * nll).sum(dim=1).mean()
    if mode == "oracle":
        k_star = nll.argmin(dim=1, keepdim=True)
        return nll.gather(1, k_star).mean() - torch.log(w.gather(1, k_star)).mean()
    raise ValueError(f"Unknown classification loss: {mode}")


def prediction_logits(mode, out, expert_logits, w):
    """Tensor to take the argmax of for prediction: for mixprob the log of the mixture probability (consistent with its loss), for the other modes the mixed logits."""
    if mode != "mixprob":
        return out
    return torch.logsumexp(torch.log(w)[:, :, None] + F.log_softmax(expert_logits.float(), dim=-1), dim=1)


def load_balance_loss(probs, topk_idx):
    """N_e * sum_i f_i * p_i, no gradient through f_i (consistent with the original script and Eq. (8) of the paper)."""
    counts = F.one_hot(topk_idx, NUM_EXPERTS).sum(dim=1).float()  # (B, N_e)
    f = counts.sum(0) / (probs.size(0) * TOP_K)
    return NUM_EXPERTS * (f.detach() * probs.mean(0)).sum()


# ======================================================================
# controller inputs
# ======================================================================
def gini_and_min(usage_pct):
    u = np.sort(np.asarray(usage_pct, dtype=np.float64)) / 100.0
    n = len(u)
    gini = (2 * np.sum(np.arange(1, n + 1) * u)) / (n * np.sum(u)) - (n + 1) / n
    return float(gini), float(np.min(usage_pct))


def balance_score(usage_pct, beta):
    """Eqs. (10)(11) of the paper: S_usage = u_min * N_e, S_gini = 200 (0.5 - G), smoothed minimum."""
    gini, u_min = gini_and_min(usage_pct)
    s_usage = u_min * NUM_EXPERTS
    s_gini = 200 * (0.5 - gini)
    m = min(s_usage, s_gini)
    bs_raw = m - (1 / beta) * math.log1p(math.exp(-beta * abs(s_usage - s_gini)))
    return bs_raw, gini, u_min


def trend(history, w):
    """Eq. (12) of the paper: T = mean_{i=1..W}(a_t - a_{t-i}). Returns 0 when the history has fewer than W+1 epochs."""
    if len(history) < w + 1:
        return 0.0
    a_t = history[-1]
    return float(np.mean([a_t - history[-1 - i] for i in range(1, w + 1)]))


# ======================================================================
# controller
# ======================================================================
class FuzzySystem:
    """Mamdani inference (skfuzzy: min/max operators, centroid defuzzification). Universes, membership functions and rules all come from cfg (JSON).
    Rules: {"name", "if": {input: level}, "op": "and" (default) | "or", "then": {output: level}}.
    When early_mode is true and the T_slow history is insufficient (the first window_slow epochs), only rules without T_slow are used (early subsystem); when no rule fires, the previous epoch's values are kept."""

    def __init__(self, cfg):
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl
        self.fuzz = fuzz
        self.cfg = cfg
        self.vars = {}

        def universe(r):
            start, stop, step = r
            return np.linspace(start, stop, int(round((stop - start) / step)) + 1)

        def mf(u, kind, params):
            return fuzz.trapmf(u, params) if kind == "trap" else fuzz.trimf(u, params)

        for name, spec in cfg["inputs"].items():
            v = ctrl.Antecedent(universe(spec["range"]), name)
            for term, (kind, params) in spec["terms"].items():
                v[term] = mf(v.universe, kind, params)
            self.vars[name] = v
        for name, spec in cfg["outputs"].items():
            v = ctrl.Consequent(universe(spec["range"]), name)  # centroid defuzzification by default
            for term, (kind, params) in spec["terms"].items():
                v[term] = mf(v.universe, kind, params)
            self.vars[name] = v
        self.rules = cfg["rules"]
        names = [r["name"] for r in self.rules]
        if len(set(names)) != len(names):
            raise SystemExit(f"[Error] Duplicate rule names in the fuzzy config: {names}")

        def build(rules):
            built = []
            for r in rules:
                conds = [self.vars[k][term] for k, term in r["if"].items()]
                ante = conds[0]
                for c in conds[1:]:
                    ante = (ante | c) if r.get("op", "and") == "or" else (ante & c)
                built.append(ctrl.Rule(ante, tuple(self.vars[k][term] for k, term in r["then"].items())))
            return ctrl.ControlSystemSimulation(ctrl.ControlSystem(built)) if built else None

        self.sim = build(self.rules)
        self.early_rules = [r for r in self.rules if "T_slow" not in r["if"]]
        self.sim_early = build(self.early_rules)
        self.early_inputs = {k for r in self.early_rules for k in r["if"]}

    def clip(self, name, value):
        lo, hi, _ = self.cfg["inputs"][name]["range"]
        return float(np.clip(value, lo, hi))

    def memberships(self, inputs):
        out = {}
        for name, value in inputs.items():
            v = self.vars[name]
            for term in self.cfg["inputs"][name]["terms"]:
                out[f"mu_{name}_{term}"] = float(self.fuzz.interp_membership(v.universe, v[term].mf, value))
        return out

    def rule_activations(self, mu, rules=None):
        acts = {}
        for r in (self.rules if rules is None else rules):
            m = [mu[f"mu_{k}_{term}"] for k, term in r["if"].items()]
            acts[r["name"]] = max(m) if r.get("op", "and") == "or" else min(m)
        return acts

    def compute(self, inputs, early=False):
        """Return (output dict or None, mode). Returns (None, "early-idle") when no rule of the early subsystem fires."""
        sim, rules = (self.sim_early, self.early_rules) if early else (self.sim, self.rules)
        if early:
            if sim is None or max(self.rule_activations(self.memberships(inputs), rules).values(), default=0.0) <= 0.0:
                return None, "early-idle"
            inputs = {k: v for k, v in inputs.items() if k in self.early_inputs}
        for name, value in inputs.items():
            sim.input[name] = value
        sim.compute()
        return {k: float(sim.output[k]) for k in ("delta_tau", "lambda_out", "noise")}, ("early" if early else "full")


class Controller:
    """Call step() at the start of every epoch; returns the τ, λ, σ used in this epoch and the log fields."""

    def __init__(self, args, fuzzy_cfg):
        self.mode = args.controller
        self.args = args
        self.cfg = fuzzy_cfg
        self.tau, self.lam, self.sigma = args.tau0, args.lambda0, args.sigma0
        if self.mode == "auxfree":
            self.lam = args.auxfree_lambda
        self.fuzzy = FuzzySystem(fuzzy_cfg) if self.mode == "fuzzy" else None

    def step(self, epoch, acc_pi_history, usage_history):
        a = self.args
        log = {}
        # all modes log the control signals for comparison; only fuzzy mode uses them for adjustment
        if usage_history:
            bs_raw, gini, u_min = balance_score(usage_history[-1], self.cfg["beta"])
            tf = trend(acc_pi_history, self.cfg["window_fast"])
            ts = trend(acc_pi_history, self.cfg["window_slow"])
            log.update({"bs_raw": bs_raw, "gini": gini, "min_usage": u_min, "t_fast_raw": tf, "t_slow_raw": ts})
        active = epoch >= a.warmup_epochs  # no adjustment during the learning-rate warmup

        if self.mode == "schedule" and active:
            span = max(a.epochs - 1 - a.warmup_epochs, 1)
            frac = min(max((epoch - a.warmup_epochs) / span, 0.0), 1.0)
            self.tau = a.tau0 + frac * (a.sched_tau_end - a.tau0)
            self.lam = a.sched_lambda_start + frac * (a.sched_lambda_end - a.sched_lambda_start)
            self.sigma = a.sched_sigma_start + frac * (a.sched_sigma_end - a.sched_sigma_start)
        elif self.mode == "schedule":
            self.tau, self.lam, self.sigma = a.tau0, a.sched_lambda_start, a.sched_sigma_start

        if self.mode == "fuzzy" and active and usage_history:
            inputs = {
                "B": self.fuzzy.clip("B", log["bs_raw"]),
                "T_fast": self.fuzzy.clip("T_fast", log["t_fast_raw"]),
                "T_slow": self.fuzzy.clip("T_slow", log["t_slow_raw"]),
            }
            log.update({"bs_clipped": inputs["B"], "t_fast_clipped": inputs["T_fast"],
                        "t_slow_clipped": inputs["T_slow"]})
            mu = self.fuzzy.memberships(inputs)
            log.update(mu)
            log.update({f"act_{k}": v for k, v in self.fuzzy.rule_activations(mu).items()})
            early = bool(self.cfg.get("early_mode")) and len(acc_pi_history) < self.cfg["window_slow"] + 1  # while T_slow has no real value yet (the code fills 0), use only rules without T_slow
            try:
                o, mode = self.fuzzy.compute(inputs, early=early)
                if o is not None:
                    self.tau = float(np.clip(self.tau + o["delta_tau"], self.cfg["tau_min"], self.cfg["tau_max"]))
                    a = float(self.cfg.get("output_ema", 0.0))  # moving-average coefficient for the λ, σ rule outputs (stage 8 round 2); 0 means taking the rule outputs directly
                    self.lam = a * self.lam + (1 - a) * o["lambda_out"]
                    self.sigma = a * self.sigma + (1 - a) * o["noise"]
                    log.update({"fuzzy_delta_tau": o["delta_tau"], "fuzzy_lambda_out": o["lambda_out"], "fuzzy_noise_out": o["noise"]})
                log.update({"fuzzy_ok": True, "fuzzy_mode": mode})
            except Exception as err:  # skfuzzy cannot defuzzify when e.g. no rule fires; keep the previous epoch's values
                log.update({"fuzzy_ok": False, "fuzzy_mode": "early" if early else "full", "fuzzy_error": str(err)[:120]})

        log.update({"tau": self.tau, "lambda": self.lam, "sigma": self.sigma, "controller_active": active})
        return log


# ======================================================================
# training and validation
# ======================================================================
@torch.no_grad()
def evaluate(model, loader, device, num_classes, pi, max_steps=0, cls_loss="mix"):
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    ps_top1, agree = [], 0  # predictions of the top-1 expert alone; number of samples on which the two selected experts agree
    per_class_expert = np.zeros((num_classes, NUM_EXPERTS), dtype=np.int64)
    per_class_weight = np.zeros((num_classes, NUM_EXPERTS), dtype=np.float64)  # per class, sum of the mixing weights w that its samples give to each expert
    per_class_nobias = np.zeros((num_classes, NUM_EXPERTS), dtype=np.int64)  # auxfree: top-k counts without bias (affinity only)
    bias_changed = 0  # auxfree: number of samples whose top-k set is changed by the bias
    gaps, w1s, topks, probs_all = [], [], [], []
    for step, (x, y, _) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out, probs, topk_idx, expert_logits = model(x)
        w = mix_weights(probs, topk_idx)
        pred = prediction_logits(cls_loss, out, expert_logits, w)
        # cosine score difference and larger weight of the two selected experts. No noise at validation: with softmax s_a - s_b = τ·log(p_a / p_b),
        # with auxfree (sigmoid affinity) s_a - s_b = τ·(logit(a_a) - logit(a_b))
        sel = probs.gather(1, topk_idx)
        hi, lo = sel.max(1).values, sel.min(1).values
        if model.use_bias:
            gaps.append((model.tau * (torch.logit(hi) - torch.logit(lo))).cpu().numpy())
            nb = torch.topk(probs, TOP_K, dim=-1).indices
            bias_changed += (nb.sort(1).values != topk_idx.sort(1).values).any(1).sum().item()
            for j in range(TOP_K):
                np.add.at(per_class_nobias, (y.cpu().numpy(), nb[:, j].cpu().numpy()), 1)
        else:
            gaps.append((model.tau * torch.log(hi / lo)).cpu().numpy())
        w1s.append((hi / (hi + lo)).cpu().numpy())
        loss_sum += classification_loss(cls_loss, out, expert_logits, w, y).item() * y.numel()
        n += y.numel()
        yc, kc = y.cpu().numpy(), topk_idx.cpu().numpy()
        wc = w.float().cpu().numpy()  # normalized mixing weights w of the selected experts
        for j in range(TOP_K):
            np.add.at(per_class_expert, (yc, kc[:, j]), 1)
            np.add.at(per_class_weight, (yc, kc[:, j]), wc[:, j])
        topks.append(kc)
        probs_all.append(probs.float().cpu().numpy().astype(np.float16))
        ys.append(yc)
        ps.append(pred.argmax(1).cpu().numpy())
        single = expert_logits.argmax(2)  # (B, K) prediction of each selected expert alone
        ps_top1.append(single[:, 0].cpu().numpy())
        agree += (single[:, 0] == single[:, -1]).sum().item()
    m = classification_metrics(np.concatenate(ys), np.concatenate(ps), num_classes, pi)
    m["loss"] = loss_sum / max(n, 1)
    m["acc_pi_top1"] = classification_metrics(np.concatenate(ys), np.concatenate(ps_top1), num_classes, pi)["acc_pi"]
    m["expert_agree"] = agree / max(n, 1)
    m["per_class_expert"] = per_class_expert
    m["usage_weighted"] = weighted_expert_usage(per_class_expert, pi, TOP_K)
    raw = per_class_expert.sum(0)
    m["usage_raw"] = raw / raw.sum() * 100
    m["weight_share"] = weighted_expert_usage(per_class_weight, pi, 1)  # share of w (%), combined by π; w sums to 1 for each sample
    if model.use_bias:
        m["usage_nobias"] = weighted_expert_usage(per_class_nobias, pi, TOP_K)
        m["bias_changed_top2"] = bias_changed / max(n, 1)
    m["cos_gap_p10"], m["cos_gap_p50"], m["cos_gap_p90"] = np.percentile(np.concatenate(gaps), [10, 50, 90]).tolist()
    m["w1_mean"] = float(np.concatenate(w1s).mean())
    m["topk"] = np.concatenate(topks)  # (num samples, TOP_K), column 0 is top-1; the validation order is fixed, so it can be compared across epochs
    m["probs"] = np.concatenate(probs_all)  # (num samples, N_e), float16
    return m


def diagnostic_groups(model):
    """Parameter groups for training diagnostics: router conv part (R0 only), proj and expert_emb of the router cosine head, shared backbone, experts. Checks that every parameter belongs to exactly one group."""
    r = model.router
    groups = {}
    if isinstance(r, Router):
        groups["router_conv"] = [p for m in (r.conv1, r.bn1, r.conv2, r.bn2) for p in m.parameters()]
    groups.update({
        "router_proj": list(r.proj.parameters()),
        "router_emb": [r.expert_emb],  # separate group: its norm is much smaller than proj, so its changes would be masked by proj if merged
        "shared": list(model.shared_extractor.parameters()),
        "experts": list(model.experts.parameters()),
    })
    ids = [id(p) for ps in groups.values() for p in ps]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()}:
        raise RuntimeError("Diagnostic parameter groups must cover every model parameter exactly once")
    return groups


def parse_args():
    ap = argparse.ArgumentParser(description="Train MoE (fixed / schedule / fuzzy / auxfree)")
    ap.add_argument("--controller", choices=["fixed", "schedule", "fuzzy", "auxfree"], required=True)
    ap.add_argument("--cls-loss", choices=["mix", "mixprob", "expert", "oracle"], default="mix",
                    help="Classification loss: mix = CE on the mixed logits (original); mixprob = -log of the mixed probabilities; "
                         "expert = gate-weighted per-expert CE; oracle = CE of the best selected expert plus -log of its gate weight")
    ap.add_argument("--router", choices=list(ROUTERS), default="R0",
                    help="R0: two 3x3 conv layers + cosine head; R4: global average pooling + cosine head (see analysis/Pilot_Plan.md stage 5)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--subset-dir", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--val-list", default="val.csv")
    ap.add_argument("--aug", choices=["A", "B"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, required=True)
    # initial values of the routing hyperparameters (used throughout in fixed mode)
    ap.add_argument("--tau0", type=float, default=0.5, help="Cosine-router temperature (Tutel/GMoE init_t=0.5)")
    ap.add_argument("--lambda0", type=float, default=1.5)
    ap.add_argument("--sigma0", type=float, default=0.1)
    ap.add_argument("--aux-warmup-frac", type=float, default=1 / 3,
                    help="Multiply the auxiliary loss weight by 0.1 for the first floor(frac*epochs + 0.5) epochs (the original script used 20 epochs)")
    # end points of schedule mode (start points: τ uses tau0, λ and σ use sched_*_start)
    # stage 8 start and end points: cover the same λ, σ range as the fuzzy outputs, τ goes from the baseline 0.5 to 0.25 (analysis/Pilot_Study.md stage 8)
    ap.add_argument("--sched-tau-end", type=float, default=0.25)
    ap.add_argument("--sched-lambda-start", type=float, default=4.5)
    ap.add_argument("--sched-lambda-end", type=float, default=0.5)
    ap.add_argument("--sched-sigma-start", type=float, default=0.8)
    ap.add_argument("--sched-sigma-end", type=float, default=0.1)
    # fuzzy mode
    ap.add_argument("--fuzzy-config", default=str(FUZZY_CONFIG_DEFAULT),
                    help="JSON with universes, membership functions, rules, windows, beta and tau bounds; 'legacy' = the submitted paper's controller")
    # auxfree mode
    ap.add_argument("--auxfree-gamma", type=float, default=1e-3, help="Routing bias update step size")
    ap.add_argument("--auxfree-lambda", type=float, default=0.0, help="Auxiliary loss weight retained in auxfree mode")
    # disturbance trial (disturb-pilot branch): add an external bias to one expert's top-k selection score midway through training, removed after the window ends
    ap.add_argument("--disturb-expert", type=int, default=-1, help="Expert whose selection score is biased during training; -1 = off")
    ap.add_argument("--disturb-epochs", default="", help="1-based inclusive epoch window for the bias, e.g. 21-25")
    ap.add_argument("--disturb-bias", type=float, default=1.0, help="Value added to the target expert's selection score (training only)")
    ap.add_argument("--stop-after-epoch", type=int, default=0,
                    help="Stop after this epoch (1-based) and finish normally; the LR schedule still follows --epochs. 0 = off")
    ap.add_argument("--num-workers", type=int, default=default_num_workers())
    ap.add_argument("--no-pretrained", action="store_true", help="For smoke tests only")
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="For smoke tests only")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--env-lock", default=str(ENV_LOCK_DEFAULT),
                    help="Environment lock file checked before running; 'none' skips the check (smoke tests only)")
    ap.add_argument("--no-diagnostics", action="store_true",
                    help="Disable per-epoch gradient/parameter diagnostics (diag_* columns); training is unaffected either way")
    return ap.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.warmup_epochs < args.epochs:
        raise SystemExit("[Error] --warmup-epochs must be in [0, epochs)")
    disturb_window = None
    if args.disturb_expert >= 0:
        if not 0 <= args.disturb_expert < NUM_EXPERTS:
            raise SystemExit(f"[Error] --disturb-expert must be in [0, {NUM_EXPERTS - 1}] or -1")
        try:
            lo, hi = (int(v) for v in args.disturb_epochs.split("-"))
        except ValueError:
            raise SystemExit("[Error] --disturb-epochs must look like 21-25 when --disturb-expert is set")
        if not 1 <= lo <= hi <= args.epochs:
            raise SystemExit("[Error] --disturb-epochs must satisfy 1 <= start <= end <= epochs")
        disturb_window = (lo, hi)
    elif args.disturb_epochs:
        raise SystemExit("[Error] --disturb-epochs requires --disturb-expert")
    if not 0 <= args.stop_after_epoch <= args.epochs:
        raise SystemExit("[Error] --stop-after-epoch must be in [0, epochs]")
    env = check_env(args.env_lock)  # before creating the output directory
    out_dir = prepare_out_dir(args.out_dir, args.overwrite)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.fuzzy_config == "legacy":
        fuzzy_cfg = json.loads(json.dumps(FUZZY_LEGACY))
    else:
        with open(args.fuzzy_config, encoding="utf-8") as f:
            fuzzy_cfg = json.load(f)
        for k in ("inputs", "outputs", "rules", "window_fast", "window_slow", "beta", "tau_min", "tau_max"):
            if k not in fuzzy_cfg:
                raise SystemExit(f"[Error] {args.fuzzy_config} lacks the field {k!r}")
        if not 0.0 <= float(fuzzy_cfg.get("output_ema", 0.0)) < 1.0:
            raise SystemExit(f"[Error] {args.fuzzy_config}: output_ema must be in [0, 1)")

    subset_dir = Path(args.subset_dir).expanduser()
    class_names, pi = load_manifest(subset_dir / "manifest.json")
    num_classes = len(class_names)
    train_tf, eval_tf = build_transforms(args.aug)
    train_ds = CSVImageDataset(args.root, subset_dir / args.train_list, class_names, train_tf)
    val_ds = CSVImageDataset(args.root, subset_dir / args.val_list, class_names, eval_tf)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, args.seed)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, args.seed)

    model = MoEModel(num_classes, pretrained=not args.no_pretrained,
                     use_bias=args.controller == "auxfree", tau=args.tau0, router=args.router).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(train_loader) if not args.max_steps_per_epoch else min(len(train_loader), args.max_steps_per_epoch)
    scheduler = build_scheduler(optimizer, steps, args.epochs, args.warmup_epochs, args.min_lr)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    controller = Controller(args, fuzzy_cfg)
    aux_warmup_epochs = math.floor(args.aux_warmup_frac * args.epochs + 0.5)

    config = {**vars(args), "env": env, "model": "moe", "class_names": class_names, "num_experts": NUM_EXPERTS,
              "top_k": TOP_K, "split_block": SPLIT_BLOCK, "aux_warmup_epochs": aux_warmup_epochs,
              "router_params": sum(p.numel() for p in model.router.parameters()),
              "fuzzy_config": fuzzy_cfg if args.controller == "fuzzy" else None,
              "train_size": len(train_ds), "val_size": len(val_ds), "device": str(device)}
    save_json(config, out_dir / "config.json")
    logger = CSVLogger(out_dir / "history.csv")
    diag = None if args.no_diagnostics else TrainDiagnostics(diagnostic_groups(model),
                                                             angle_params={"router_emb": model.router.expert_emb})

    acc_pi_history, usage_history = [], []
    prev_topk = None  # validation top-k of the previous epoch, for the routing-assignment fluctuation diagnostic
    route_topk, route_probs = [], []  # per-sample validation routing records of each epoch, written to val_routing.npz at the end
    best = {"acc_pi": -1.0, "epoch": -1}
    t_start = time.time()
    for epoch in range(args.epochs):
        clog = controller.step(epoch, acc_pi_history, usage_history)
        model.tau.fill_(controller.tau)
        model.router.noise_scale.fill_(controller.sigma)
        aux_factor = 0.1 if epoch < aux_warmup_epochs else 1.0
        lam_eff = controller.lam * aux_factor
        disturb_active = disturb_window is not None and disturb_window[0] <= epoch + 1 <= disturb_window[1]
        if disturb_window is not None:
            model.disturb_on = disturb_active
            model.disturb_bias.zero_()
            if disturb_active:
                model.disturb_bias[args.disturb_expert] = args.disturb_bias

        model.train()
        if diag:
            diag.start_epoch(scaler)
        t0 = time.time()
        cls_sum, aux_sum, correct, n = 0.0, 0.0, 0, 0
        train_counts = np.zeros(NUM_EXPERTS, dtype=np.int64)
        train_counts_nobias = np.zeros(NUM_EXPERTS, dtype=np.int64)  # auxfree: top-k counts without bias
        for step, (x, y, _) in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out, probs, topk_idx, expert_logits = model(x)
                w = mix_weights(probs, topk_idx)
                cls_loss = classification_loss(args.cls_loss, out, expert_logits, w, y)
                aux_loss = load_balance_loss(probs, topk_idx)
                loss = cls_loss + lam_eff * aux_loss
            scaler.scale(loss).backward()
            if diag:
                diag.after_backward()
            scaler.step(optimizer)
            scaler.update()
            if diag:
                diag.after_step(scaler)
            scheduler.step()

            batch_counts = torch.bincount(topk_idx.flatten(), minlength=NUM_EXPERTS)
            if args.controller == "auxfree":  # overloaded: lower the bias; underloaded: raise the bias
                with torch.no_grad():
                    load = batch_counts.float() / topk_idx.numel()
                    model.route_bias -= args.auxfree_gamma * torch.sign(load - 1.0 / NUM_EXPERTS)
                    nb = torch.topk(probs.detach(), TOP_K, dim=-1).indices
                    train_counts_nobias += torch.bincount(nb.flatten(), minlength=NUM_EXPERTS).cpu().numpy()
            train_counts += batch_counts.cpu().numpy()
            cls_sum += cls_loss.item() * y.numel()
            aux_sum += aux_loss.item() * y.numel()
            pred = prediction_logits(args.cls_loss, out.detach(), expert_logits.detach(), w.detach())
            correct += (pred.argmax(1) == y).sum().item()
            n += y.numel()
        train_time = time.time() - t0
        diag_row = diag.end_epoch() if diag else {}

        t1 = time.time()
        vm = evaluate(model, val_loader, device, num_classes, pi, args.max_steps_per_epoch, args.cls_loss)
        if diag:  # routing-assignment fluctuation: fraction of validation samples whose top-1 / top-2 (as a set) changed compared with the previous epoch
            topk = vm["topk"]
            if prev_topk is not None and prev_topk.shape == topk.shape:
                diag_row["diag_val_route_change_top1"] = float(np.mean(topk[:, 0] != prev_topk[:, 0]))
                diag_row["diag_val_route_change_top2"] = float(np.mean(np.any(np.sort(topk, 1) != np.sort(prev_topk, 1), axis=1)))
            prev_topk = topk
            route_topk.append(topk.astype(np.int8))
            diag_row["diag_val_weight_share"] = np.round(vm["weight_share"], 3).tolist()
            diag_row["diag_val_weight_share_min"] = float(np.min(vm["weight_share"]))
            route_probs.append(vm["probs"])
        val_time = time.time() - t1
        acc_pi_history.append(vm["acc_pi"])
        usage_history.append(vm["usage_weighted"])

        is_best = vm["acc_pi"] > best["acc_pi"]
        if is_best:
            best = {"acc_pi": vm["acc_pi"], "epoch": epoch + 1}
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "val_metrics": {k: vm[k] for k in ("acc", "acc_pi", "balanced_acc", "macro_f1")},
                        "config": config}, out_dir / "best.pth")

        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
               "aux_factor": aux_factor, "lambda_eff": lam_eff, **clog,
               "train_cls_loss": cls_sum / max(n, 1), "train_aux_loss": aux_sum / max(n, 1),
               "train_acc": correct / max(n, 1),
               "train_usage": (train_counts / max(train_counts.sum(), 1) * 100).round(3).tolist(),
               "val_loss": vm["loss"], "val_acc": vm["acc"], "val_acc_pi": vm["acc_pi"],
               "val_balanced_acc": vm["balanced_acc"], "val_macro_f1": vm["macro_f1"],
               "val_acc_pi_top1": vm["acc_pi_top1"], "val_expert_agree": vm["expert_agree"],
               "val_usage_weighted": np.round(vm["usage_weighted"], 3).tolist(),
               "val_usage_raw": np.round(vm["usage_raw"], 3).tolist(),
               "val_cos_gap_p10": vm["cos_gap_p10"], "val_cos_gap_p50": vm["cos_gap_p50"],
               "val_cos_gap_p90": vm["cos_gap_p90"], "val_w1_mean": vm["w1_mean"],
               "val_per_class_recall": vm["per_class"]["recall"],
               "route_bias": model.route_bias.cpu().numpy().round(5).tolist(),
               "is_best": is_best, "train_time_s": round(train_time, 1), "val_time_s": round(val_time, 1),
               **diag_row}
        if args.controller == "auxfree":  # columns added only for auxfree; history columns of the other modes are unchanged
            row.update({"train_usage_nobias": (train_counts_nobias / max(train_counts_nobias.sum(), 1) * 100).round(3).tolist(),
                        "val_usage_nobias": np.round(vm["usage_nobias"], 3).tolist(),
                        "val_bias_changed_top2": vm["bias_changed_top2"]})
        if disturb_window is not None:
            row.update({"disturb_active": disturb_active, "disturb_target": args.disturb_expert})
        logger.log(row)
        print(f"[{epoch + 1}/{args.epochs}] τ={controller.tau:.3f} λ={lam_eff:.3f} σ={controller.sigma:.3f} "
              f"lr={row['lr']:.2e} | train acc={row['train_acc']:.4f} | val acc={vm['acc']:.4f} "
              f"A_pi={vm['acc_pi']:.4f} BA={vm['balanced_acc']:.4f} F1={vm['macro_f1']:.4f} "
              f"{'*' if is_best else ''} | usage={np.round(vm['usage_weighted'], 1).tolist()} "
              f"| {train_time:.0f}s + {val_time:.0f}s")
        if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch:
            print(f"[Info] --stop-after-epoch {args.stop_after_epoch}: stopping after epoch {epoch + 1}")
            break

    if diag and route_topk:  # (epoch, sample, TOP_K) and (epoch, sample, N_e); sample order matches the val list
        np.savez_compressed(out_dir / "val_routing.npz", topk=np.stack(route_topk), probs=np.stack(route_probs),
                            epochs=np.arange(1, len(route_topk) + 1))
    last_epoch = len(acc_pi_history)
    torch.save({"model_state_dict": model.state_dict(), "epoch": last_epoch, "config": config},
               out_dir / "last.pth")
    summary = {"best_epoch": best["epoch"], "best_val_acc_pi": best["acc_pi"],
               "total_time_min": round((time.time() - t_start) / 60, 1)}
    if last_epoch != args.epochs:
        summary["stopped_after_epoch"] = last_epoch
    save_json(summary, out_dir / "summary.json")
    print(f"Done. Best validation A_pi={best['acc_pi']:.4f} (epoch {best['epoch']}), Output directory: {out_dir}")


if __name__ == "__main__":
    main()