"""Deep Q-Network for the driver dynamic clustering agent (paper Sec. V-A, Alg. 1).

"Determining the number of clusters involves a discrete action space and continuous
state space, and thus the deep Q-network (DQN) algorithm is utilized to solve it."

Sec. VI-A fixes the network settings: learning rate 0.001, discount 0.9, memory
capacity 2000, batch size 32, target network updated every 200 rounds, and "a
three-layer small neural network". Implemented on NumPy so the reproduction carries no
deep-learning dependency; the network is small enough that this costs nothing.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class MLP:
    """Three layers: input -> hidden (ReLU) -> output, trained with Adam."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, lr: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.w1 = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.w2 = rng.normal(0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self._params = [self.w1, self.b1, self.w2, self.b2]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._step = 0

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        return h, h @ self.w2 + self.b2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(np.atleast_2d(x))[1]

    def train_step(self, x: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        """One gradient step of the squared TD error on the taken actions."""
        h, q = self.forward(x)
        err = np.zeros_like(q)
        rows = np.arange(len(x))
        err[rows, actions] = (q[rows, actions] - targets) / len(x)

        gw2 = h.T @ err
        gb2 = err.sum(0)
        dh = (err @ self.w2.T) * (h > 0)
        gw1 = x.T @ dh
        gb1 = dh.sum(0)

        self._step += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, g in enumerate([gw1, gb1, gw2, gb2]):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g * g
            self._params[i] -= self.lr * (self._m[i] / (1 - b1 ** self._step)) / (
                np.sqrt(self._v[i] / (1 - b2 ** self._step)) + eps
            )
        return float(np.mean((q[rows, actions] - targets) ** 2))

    def copy_from(self, other: "MLP") -> None:
        for p, q in zip(self._params, other._params):
            p[...] = q


class DQNAgent:
    """epsilon-greedy DQN over a discrete action set (Algorithm 1)."""

    def __init__(
        self,
        n_state: int,
        actions: tuple[int, ...],
        lr: float = 1e-3,
        gamma: float = 0.9,
        memory: int = 2000,
        batch: int = 32,
        target_update: int = 200,
        hidden: int = 64,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 2000,
        seed: int = 0,
    ):
        self.actions = actions
        self.gamma = gamma
        self.batch = batch
        self.target_update = target_update
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.rng = np.random.default_rng(seed)
        self.q = MLP(n_state, hidden, len(actions), lr, seed)
        self.target = MLP(n_state, hidden, len(actions), lr, seed)
        self.target.copy_from(self.q)
        self.memory: deque = deque(maxlen=memory)
        self.steps = 0
        self.training = True

    @property
    def epsilon(self) -> float:
        if not self.training:
            return 0.0
        frac = min(1.0, self.steps / max(self.eps_decay, 1))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def act(self, state: np.ndarray) -> int:
        """Return the *number of clusters* for this time slot."""
        self.steps += 1
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.actions)))
        return int(np.argmax(self.q(state)[0]))

    def remember(self, s, a, r, s2, done: bool) -> None:
        self.memory.append((np.asarray(s, float), a, float(r), np.asarray(s2, float), done))

    def learn(self) -> float | None:
        """Algorithm 1, lines 10-13."""
        if len(self.memory) < self.batch:
            return None
        idx = self.rng.choice(len(self.memory), size=self.batch, replace=False)
        s, a, r, s2, done = zip(*[self.memory[i] for i in idx])
        s = np.stack(s)
        s2 = np.stack(s2)
        a = np.array(a, dtype=int)
        r = np.array(r, dtype=float)
        done = np.array(done, dtype=bool)

        # y_k = r_k                              if terminal
        #       r_k + gamma_0 * max_a' Qhat(...)  otherwise
        next_q = self.target(s2).max(axis=1)
        targets = np.where(done, r, r + self.gamma * next_q)
        loss = self.q.train_step(s, a, targets)

        if self.steps % self.target_update == 0:
            self.target.copy_from(self.q)
        return loss
