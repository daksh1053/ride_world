"""Proposed method: Multi-Objective Multi-Agent Q-Learning (paper Sec. 4.3, 4.4).

MDP (Sec. 4.3)
    State   `g_v^t`, the node the driver currently occupies.
    Action  assignment of an incoming request, characterised by its OD pair
            `(s_a, d_a)`; "no action at t" is always available.
    Reward  Eq. 6, `r_{s,A}^t(v) = sum_{a in A_v^t} Geo(s_a, d_a) - Geo(g_v^t, s_a)`.

Multi-objective (Sec. 4.4). Each agent keeps a *vector* value function with one
component per objective — utility and fairness — and a scalarisation function
reduces it to a scalar for action selection:

    SR(M) = sum_v r_{s,A}(v) - lambda * omega * Var(r_{s,A}(v))          (Eq. 7)

A centralised controller (Sec. 4, "Fairness and Equity Considerations") sees all
agents and resolves each round with a max-weight matching over the scalarised
values, so allocation decisions account for the collective state of the fleet.

Prediction (Sec. 4.2, Fig. 3). During training the action space is augmented with
*predicted* future requests, so the Q-tables are updated on the pattern of requests
that will be raised later in the horizon rather than only on history. Setting
`use_prediction=False` yields the "Our Method w/o Prediction" ablation of Table 2;
setting `use_fairness=False` yields "Our Method w/o Fairness".
"""

from __future__ import annotations

import numpy as np

from .base import Matcher, hungarian_round, scaled_fairness_penalty


class MOMAQL(Matcher):
    name = "Proposed Method"

    def __init__(
        self,
        n_drivers: int,
        n_nodes: int,
        lam: float = 1.0,
        omega: float = 0.6,
        gamma: float = 0.9,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        use_prediction: bool = True,
        use_fairness: bool = True,
        per_agent_q: bool = False,
        seed: int = 0,
    ):
        self.per_agent_q = per_agent_q
        self.n_drivers = n_drivers
        self.n_nodes = n_nodes
        self.lam = lam
        self.omega = omega
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.use_prediction = use_prediction
        self.use_fairness = use_fairness
        self.rng = np.random.default_rng(seed)
        self.training = False
        self.reset()

    def reset(self) -> None:
        # One value table per objective (the "multi-objective" part): Q_u tracks
        # accumulated utility, Q_f tracks the fairness cost, both indexed by
        # (agent, current node, destination node of the action).
        #
        # ASSUMPTION: by default the agents *share* the tables (`per_agent_q=False`),
        # the usual parameter-sharing setup for a homogeneous fleet. Agents still
        # "learn different behaviours based on their starting locations" (Sec. 4)
        # because the policy is a function of the agent's own state g_v^t, and they
        # are still differentiated by the fairness component, which reads each
        # agent's own accumulated utility. Keeping one table per agent instead
        # (`per_agent_q=True`) divides the already-thin update count for the
        # 69x69 action space by the fleet size and leaves the lookahead dominated
        # by estimation noise.
        n_agents = self.n_drivers if self.per_agent_q else 1
        shape = (n_agents, self.n_nodes, self.n_nodes)
        self.q_util = np.zeros(shape, dtype=np.float64)
        self.q_fair = np.zeros(shape, dtype=np.float64)

    def _agent(self, v: int | np.ndarray):
        """Row of the Q-tables used by driver `v`."""
        return v if self.per_agent_q else np.zeros_like(v)

    # ------------------------------------------------------------- scalarisation
    def _fair_penalty(self, utilities: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """Fairness component of the vector reward; see `scaled_fairness_penalty`."""
        if not self.use_fairness:
            return np.zeros_like(gains)
        return scaled_fairness_penalty(utilities, gains, self.lam, self.omega)

    def scalarise(self, utility_gain: np.ndarray, fair_cost: np.ndarray) -> np.ndarray:
        """Eq. 7 applied to the vector-valued action values."""
        return utility_gain - fair_cost

    # ---------------------------------------------------------------- assignment
    def _greedy_next(self) -> tuple[np.ndarray, np.ndarray]:
        """Value of the scalarised-greedy continuation from every state.

        Sec. 4.4: a single-policy multi-objective algorithm "exploits scalarisation
        functions over the vector-based reward functions, thereby reducing the
        multi-objective environment's dimensionality to a single, scalar dimension".
        The greedy next action is therefore the one that maximises the *scalarised*
        value; both objective components are then read off that same action, rather
        than each objective being optimised independently.
        """
        best = np.argmax(self.scalarise(self.q_util, self.q_fair), axis=2)  # (a, node)
        a_idx = np.arange(self.q_util.shape[0])[:, None]
        n_idx = np.arange(self.n_nodes)[None, :]
        return self.q_util[a_idx, n_idx, best], self.q_fair[a_idx, n_idx, best]

    def _action_values(self, env, batch) -> tuple[np.ndarray, np.ndarray]:
        """Vector action values `(utility component, fairness component)`."""
        util = env.utility_matrix(batch)                            # (m, n)
        rows = self._agent(np.arange(self.n_drivers))               # (n,)

        # Bellman lookahead from the node the trip ends at.
        next_u, next_f = self._greedy_next()                        # (a, node) each
        future_u = self.gamma * next_u[rows[None, :], batch.dest[:, None]]
        future_f = self.gamma * next_f[rows[None, :], batch.dest[:, None]]

        q_u = util + future_u
        q_f = self._fair_penalty(env.fleet.utility, util) + future_f
        return q_u, q_f

    def assign_round(self, env, batch) -> np.ndarray:
        m = len(batch)
        if m == 0:
            return np.full(0, -1, dtype=int)

        q_u, q_f = self._action_values(env, batch)
        score = self.scalarise(q_u, q_f)

        if self.training and self.epsilon > 0:
            noise = self.rng.normal(0.0, self.epsilon * (np.abs(score).mean() + 1e-9), size=score.shape)
            score = score + noise

        # "no action at t" — never take an assignment whose scalarised value is
        # negative, which is what lets drivers idle instead of being overburdened.
        accept = (score > 0.0) & env.feasible_matrix(batch)
        return hungarian_round(score, env.available_mask(), accept=accept)

    # ------------------------------------------------------------------ learning
    def observe(self, env, batch, decision, gained, prev) -> None:
        if not self.training:
            return
        taken = np.flatnonzero(decision >= 0)
        if taken.size == 0:
            return
        utility_before = prev["utility"]
        util_gain = env.fleet.utility - utility_before
        fair_cost = self._fair_penalty(utility_before, util_gain[None, :])[0]
        next_u, next_f = self._greedy_next()

        for i in taken:
            v = int(decision[i])
            a = int(self._agent(v))
            g = int(prev["location"][v])   # state s_t: where the driver was
            d = int(batch.dest[i])
            self._update(a, g, d, util_gain[v], fair_cost[v], next_u, next_f)

    def _update(self, a: int, g: int, d: int, gain: float, fair: float,
                next_u: np.ndarray, next_f: np.ndarray) -> None:
        """One vector-valued Q-learning update on the pair (state g, action -> d)."""
        target_u = gain + self.gamma * next_u[a, d]
        target_f = fair + self.gamma * next_f[a, d]
        self.q_util[a, g, d] += self.alpha * (target_u - self.q_util[a, g, d])
        self.q_fair[a, g, d] += self.alpha * (target_f - self.q_fair[a, g, d])

    # ------------------------------------------------------------------ interface
    @property
    def wants_prediction(self) -> bool:
        """Whether the trainer should replay forecast request days into this agent.

        The look-ahead of Sec. 4.2 is delivered by `ltf.experiment.train`, which
        materialises the forecast into synthetic request days and drives them
        through exactly the same `assign_round` / `observe` path as real days — so
        the predicted requests genuinely enter the action space, and the Q-tables
        see the (state, action) distribution the controller would actually induce.
        """
        return self.use_prediction
