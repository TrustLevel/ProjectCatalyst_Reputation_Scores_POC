"""
rex-calculation.py

Advanced REX framework (post-MVP):
- Bayesian reputation (Beta posterior per agent × domain)
- Per-review expertise / context-fit
- Per-review trust (combining reliability & expertise)
- Proposal aggregation (quality + risk channels)

No external dependencies (only standard library).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import math
import time


# ---------------------------------------------------------------------------
# Basic Beta Posterior Utilities
# ---------------------------------------------------------------------------

@dataclass
class BetaState:
    """Beta(α, β) posterior for a probability in [0,1]."""
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        denom = self.alpha + self.beta
        if denom <= 0:
            return 0.5
        return self.alpha / denom

    @property
    def n(self) -> float:
        """Concentration / effective sample size."""
        return self.alpha + self.beta

    def add_evidence(self, value: float, strength: float) -> None:
        """
        Update with evidence (value ∈ [0,1], strength ≥ 0).
        This corresponds to: α += strength * value, β += strength * (1 - value)
        """
        v = max(0.0, min(1.0, value))
        s = max(0.0, strength)
        self.alpha += s * v
        self.beta += s * (1.0 - v)

    def ci95(self) -> Tuple[float, float]:
        """
        Approximate 95% credible interval using normal approximation.
        For production, you may want a more precise method.
        """
        n = self.alpha + self.beta
        if n <= 0:
            return 0.0, 1.0
        p = self.mean
        # Normal approximation to Beta
        var = p * (1.0 - p) / (n + 1.0)
        std = math.sqrt(max(var, 0.0))
        z = 1.96
        lo = max(0.0, p - z * std)
        hi = min(1.0, p + z * std)
        return lo, hi


# ---------------------------------------------------------------------------
# Evidence Model for Reputation
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """
    Generic evidential update for reputation.

    value:   in [0,1], where 1 = strong positive signal, 0 = strong negative
    strength: λ ≥ 0, evidential weight
    independence: ∈ [0,1], down-weights correlated/COI signals
    source: type of evidence, e.g. 'peer-eval', 'flag-confirmed', 'miss', 'outcome'
    """
    agent_id: str
    domain_id: str
    value: float
    strength: float
    source: str
    timestamp: float = field(default_factory=lambda: time.time())
    independence: float = 1.0
    metadata: Dict[str, str] = field(default_factory=dict)

    def effective_strength(self) -> float:
        """Apply independence factor (and any other simple scaling)."""
        indep = max(0.0, min(1.0, self.independence))
        s = max(0.0, self.strength)
        return s * indep


# ---------------------------------------------------------------------------
# Reputation Store (Reliability R per agent × domain)
# ---------------------------------------------------------------------------

@dataclass
class ReputationConfig:
    """Hyperparameters for prior and evidence scaling."""
    prior_mean: float = 0.5   # m0
    prior_strength: float = 2.0  # ν (small → neutral but uncertain)
    # Multipliers per evidence source (optional, for tuning)
    scale_peer_eval: float = 1.0
    scale_flag_confirmed: float = 1.0
    scale_miss: float = 1.0
    scale_outcome: float = 0.5
    scale_impact: float = 0.25


class ReputationStore:
    """
    Stores and updates Beta(α,β) states per (agent_id, domain_id).
    """

    def __init__(self, config: Optional[ReputationConfig] = None):
        self.config = config or ReputationConfig()
        self._states: Dict[Tuple[str, str], BetaState] = {}

    def _prior_state(self) -> BetaState:
        c = self.config
        alpha0 = c.prior_mean * c.prior_strength
        beta0 = (1.0 - c.prior_mean) * c.prior_strength
        return BetaState(alpha=alpha0, beta=beta0)

    def get_state(self, agent_id: str, domain_id: str) -> BetaState:
        key = (agent_id, domain_id)
        if key not in self._states:
            self._states[key] = self._prior_state()
        return self._states[key]

    def update_with_evidence(self, ev: Evidence) -> None:
        state = self.get_state(ev.agent_id, ev.domain_id)
        v = max(0.0, min(1.0, ev.value))
        lam = ev.effective_strength() * self._source_scale(ev.source)
        state.add_evidence(v, lam)

    def bulk_update(self, evidences: List[Evidence]) -> None:
        for ev in evidences:
            self.update_with_evidence(ev)

    def _source_scale(self, source: str) -> float:
        c = self.config
        if source == "peer-eval":
            return c.scale_peer_eval
        if source == "flag-confirmed":
            return c.scale_flag_confirmed
        if source == "miss":
            return c.scale_miss
        if source == "outcome":
            return c.scale_outcome
        if source == "impact":
            return c.scale_impact
        return 1.0

    def get_reliability(self, agent_id: str, domain_id: str) -> Tuple[float, float, Tuple[float, float]]:
        """
        Returns (mean R, n, (ci_lo, ci_hi)) for a given agent × domain.
        """
        state = self.get_state(agent_id, domain_id)
        return state.mean, state.n, state.ci95()


# ---------------------------------------------------------------------------
# Expertise / Context-Fit per Review
# ---------------------------------------------------------------------------

@dataclass
class ExpertiseConfig:
    """Weights for expertise components."""
    w_self: float = 0.5     # S: self-assessment
    w_validation: float = 1.0   # V: validated expertise (admin/endorsement)
    w_confidence: float = 0.4   # C: declared confidence
    w_match: float = 0.7   # M: domain/tag match
    # base intensity for λ_E
    base_lambda_E: float = 0.7


@dataclass
class ExpertiseInputs:
    """Inputs for computing expertise E_i for a given review."""
    self_assessment: float  # S ∈ [0,1]
    validation: float       # V ∈ [0,1] (0 = unvalidated, 1 = strongly validated)
    confidence: float       # C ∈ [0,1]
    match: float            # M ∈ [0,1]


@dataclass
class ExpertiseResult:
    E: float        # expertise/context-fit ∈ [0,1]
    lambda_E: float # virtual evidence strength ≥ 0


class ExpertiseModel:
    """
    Computes per-review expertise/context-fit E_i and its evidential strength λ_E.
    """

    def __init__(self, config: Optional[ExpertiseConfig] = None):
        self.config = config or ExpertiseConfig()

    def compute(self, inputs: ExpertiseInputs) -> ExpertiseResult:
        c = self.config
        S = self._clip01(inputs.self_assessment)
        V = self._clip01(inputs.validation)
        C = self._clip01(inputs.confidence)
        M = self._clip01(inputs.match)

        num = c.w_self * S + c.w_validation * V + c.w_confidence * C + c.w_match * M
        den = c.w_self + c.w_validation + c.w_confidence + c.w_match
        if den <= 0:
            E = 0.5
        else:
            E = num / den

        # λ_E grows with validation (if validated, evidence counts more)
        lambda_E = c.base_lambda_E * (0.5 + 0.5 * V)
        return ExpertiseResult(E=E, lambda_E=lambda_E)

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# Per-Review Trust (Combining R & E)
# ---------------------------------------------------------------------------

@dataclass
class ReviewTrust:
    """Per-review trust posterior."""
    trust: float   # T_i = mean of Beta(α_i, β_i)
    n: float       # n_i = α_i + β_i
    ci95: Tuple[float, float]
    alpha: float
    beta: float


def compute_review_trust(
    rep_state: BetaState,
    E_result: ExpertiseResult
) -> ReviewTrust:
    """
    Combine reputation Beta(α,β) with per-review expertise E, λ_E via virtual evidence.

    α_i = α + λ_E * E
    β_i = β + λ_E * (1 - E)
    """
    E = max(0.0, min(1.0, E_result.E))
    lam_E = max(0.0, E_result.lambda_E)

    alpha_i = rep_state.alpha + lam_E * E
    beta_i = rep_state.beta + lam_E * (1.0 - E)

    tmp = BetaState(alpha=alpha_i, beta=beta_i)
    return ReviewTrust(
        trust=tmp.mean,
        n=tmp.n,
        ci95=tmp.ci95(),
        alpha=alpha_i,
        beta=beta_i,
    )


# ---------------------------------------------------------------------------
# Proposal Aggregation: Quality & Risk
# ---------------------------------------------------------------------------

@dataclass
class ProposalConfig:
    """Hyperparameters for proposal aggregation."""
    # Prior for quality posterior
    quality_prior_mean: float = 0.5
    quality_prior_strength: float = 1.0
    # prior for risk posterior (probability of being low-quality/problematic)
    risk_prior_mean: float = 0.1
    risk_prior_strength: float = 1.0

    # Saturation for trust mass: g(n_i) = n_i / (n_i + c)
    saturation_c: float = 5.0

    # Global scales
    kappa_quality: float = 1.0
    kappa_flag: float = 1.0

    # Thresholds for status
    high_quality_threshold: float = 0.7
    low_risk_threshold: float = 0.2
    red_risk_threshold: float = 0.5
    max_ci_width_for_high: float = 0.2


@dataclass
class ProposalState:
    """Holds quality and risk Beta states for a proposal."""
    quality: BetaState
    risk: BetaState


@dataclass
class ProposalSummary:
    """Human- and machine-friendly summary of a proposal's posterior states."""
    quality_mean: float
    quality_ci95: Tuple[float, float]
    quality_n: float
    risk_mean: float
    risk_ci95: Tuple[float, float]
    risk_n: float
    status: str              # "HIGH", "GREY", "RED"
    reasons: List[str]


class ProposalStore:
    """
    Stores Beta states for proposals (quality + risk) and updates them
    based on review observations and flags.
    """

    def __init__(self, config: Optional[ProposalConfig] = None):
        self.config = config or ProposalConfig()
        self._states: Dict[str, ProposalState] = {}

    def _prior_quality_state(self) -> BetaState:
        c = self.config
        alpha0 = c.quality_prior_mean * c.quality_prior_strength
        beta0 = (1.0 - c.quality_prior_mean) * c.quality_prior_strength
        return BetaState(alpha=alpha0, beta=beta0)

    def _prior_risk_state(self) -> BetaState:
        c = self.config
        alpha0 = c.risk_prior_mean * c.risk_prior_strength
        beta0 = (1.0 - c.risk_prior_mean) * c.risk_prior_strength
        return BetaState(alpha=alpha0, beta=beta0)

    def _get_state(self, proposal_id: str) -> ProposalState:
        if proposal_id not in self._states:
            self._states[proposal_id] = ProposalState(
                quality=self._prior_quality_state(),
                risk=self._prior_risk_state(),
            )
        return self._states[proposal_id]

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def observe_review(
        self,
        proposal_id: str,
        score01: float,        # y_i ∈ [0,1] (normalized review score)
        review_trust: ReviewTrust,
        flagged: bool,
    ) -> ProposalSummary:
        """
        Update the proposal's quality and risk posteriors based on a single review.
        Returns updated summary.
        """
        c = self.config
        st = self._get_state(proposal_id)

        # --- Quality update ---
        y = self._clip01(score01)
        T_i = self._clip01(review_trust.trust)
        n_i = max(0.0, review_trust.n)

        # effective weight: k_i = κ_Q * T_i * g(n_i)
        g = n_i / (n_i + c.saturation_c) if (n_i + c.saturation_c) > 0 else 0.0
        k_i = c.kappa_quality * T_i * g

        st.quality.add_evidence(y, k_i)

        # --- Risk update (flags) ---
        if flagged:
            # Interpret risk as probability "proposal is low quality / problematic"
            # A flag is positive evidence for risk.
            lam_flag = c.kappa_flag * T_i
            st.risk.add_evidence(1.0, lam_flag)

        return self.summarize(proposal_id)

    def summarize(self, proposal_id: str) -> ProposalSummary:
        """
        Compute quality/risk summaries and derive a status label.
        """
        c = self.config
        st = self._get_state(proposal_id)

        q_mean = st.quality.mean
        q_ci_lo, q_ci_hi = st.quality.ci95()
        q_n = st.quality.n

        r_mean = st.risk.mean
        r_ci_lo, r_ci_hi = st.risk.ci95()
        r_n = st.risk.n

        status, reasons = self._derive_status(
            q_mean, (q_ci_lo, q_ci_hi), r_mean, (r_ci_lo, r_ci_hi)
        )

        return ProposalSummary(
            quality_mean=q_mean,
            quality_ci95=(q_ci_lo, q_ci_hi),
            quality_n=q_n,
            risk_mean=r_mean,
            risk_ci95=(r_ci_lo, r_ci_hi),
            risk_n=r_n,
            status=status,
            reasons=reasons,
        )

    def _derive_status(
        self,
        q_mean: float,
        q_ci: Tuple[float, float],
        r_mean: float,
        r_ci: Tuple[float, float],
    ) -> Tuple[str, List[str]]:
        """
        Simple status logic based on quality & risk posterior.
        """
        c = self.config
        reasons: List[str] = []

        q_lo, q_hi = q_ci
        r_lo, r_hi = r_ci
        ci_width = q_hi - q_lo

        # High: high mean quality, low risk, narrow CI
        if (
            q_mean >= c.high_quality_threshold
            and r_mean <= c.low_risk_threshold
            and ci_width <= c.max_ci_width_for_high
        ):
            status = "HIGH"
            if q_mean < 0.8:
                reasons.append("good but not extremely high quality")
            if ci_width > 0.1:
                reasons.append("moderate uncertainty in quality")
            if r_mean > 0.1:
                reasons.append("non-zero residual risk")
            return status, reasons

        # RED: clearly risky
        if r_mean >= c.red_risk_threshold:
            status = "RED"
            reasons.append("risk posterior is high (many/credible flags)")
            if q_mean < 0.6:
                reasons.append("quality posterior not strong enough to offset risk")
            return status, reasons

        # Otherwise: GREY
        status = "GREY"
        reasons.append("more evidence needed")
        if ci_width > c.max_ci_width_for_high:
            reasons.append("quality confidence interval is wide")
        if r_mean > c.low_risk_threshold:
            reasons.append("risk not negligible yet")
        return status, reasons


# ---------------------------------------------------------------------------
# Example usage (for reference / tests)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: simple end-to-end flow for one review.

    # 1) Setup stores/models
    rep_store = ReputationStore()
    exp_model = ExpertiseModel()
    prop_store = ProposalStore()

    agent = "reviewer_1"
    domain = "ai"
    proposal = "proposal_42"

    # 2) Current reliability state (before any evidence)
    R_mean, R_n, R_ci = rep_store.get_reliability(agent, domain)
    print("Initial R:", R_mean, "n=", R_n, "CI=", R_ci)

    # 3) Expertise for this review
    exp_inputs = ExpertiseInputs(
        self_assessment=0.8,
        validation=0.5,   # partially validated
        confidence=0.9,
        match=0.7,
    )
    exp_res = exp_model.compute(exp_inputs)
    print("Expertise E:", exp_res.E, "lambda_E=", exp_res.lambda_E)

    # 4) Combine to per-review trust
    rep_state = rep_store.get_state(agent, domain)
    review_trust = compute_review_trust(rep_state, exp_res)
    print("Review trust:", review_trust.trust, "n_i=", review_trust.n, "CI=", review_trust.ci95)

    # 5) Assume a normalized review score y ∈ [0,1] and a flag
    score01 = 0.83
    flagged = True

    summary = prop_store.observe_review(
        proposal_id=proposal,
        score01=score01,
        review_trust=review_trust,
        flagged=flagged,
    )

    print("Proposal summary after one review:")
    print(summary)

    # 6) Later: Peer-eval evidence + confirmed flag update R
    ev_peer = Evidence(
        agent_id=agent,
        domain_id=domain,
        value=0.9,   # strong positive peer evaluation
        strength=1.0,
        source="peer-eval",
        independence=0.9,
    )
    ev_flag = Evidence(
        agent_id=agent,
        domain_id=domain,
        value=1.0,   # confirmed correct flag
        strength=0.7,
        source="flag-confirmed",
        independence=1.0,
    )
    rep_store.bulk_update([ev_peer, ev_flag])

    R_mean2, R_n2, R_ci2 = rep_store.get_reliability(agent, domain)
    print("Updated R:", R_mean2, "n=", R_n2, "CI=", R_ci2)