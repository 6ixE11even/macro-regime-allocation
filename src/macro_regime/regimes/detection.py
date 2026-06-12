"""
Two-step macro-regime detection.

Pipeline (all fit on the TRAIN split only — no peeking at the test years):

    standardise -> PCA(95% var) -> KMeans(k=2): crisis vs "typical"
                                -> KMeans(k*) on the typical months: the sub-regimes

The crisis cluster is whichever of the two is smaller (deep negative shocks are
rare). The number of sub-regimes k* is chosen by silhouette over k=3..7. Out of
sample we don't hard-assign — we produce soft probabilities and take the argmax,
so a month that looks half-crisis is reported as such.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, normalize

from macro_regime.config import RANDOM_STATE

logger = logging.getLogger(__name__)


def _softmax_from_score(score: np.ndarray, temperature: float) -> np.ndarray:
    """exp(-score / T), row-normalised. `score` is a distance in step 1 and a
    similarity in step 2 — kept identical to the original formulation so the
    regime probabilities reproduce exactly."""
    weights = np.exp(-score / temperature)
    return weights / weights.sum(axis=1, keepdims=True)


class RegimeDetector:
    def __init__(
        self,
        pca_variance: float = 0.95,
        k_min: int = 3,
        k_max: int = 7,
        crisis_temperature: float = 10.0,
        regime_temperature: float = 0.5,
        random_state: int = RANDOM_STATE,
    ):
        self.pca_variance = pca_variance
        self.k_min, self.k_max = k_min, k_max
        self.crisis_temperature = crisis_temperature
        self.regime_temperature = regime_temperature
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        """Learn the regimes on the training panel.

        Returns (hard regime label per training month, PCA scores per month).
        """
        self.scaler_ = StandardScaler().fit(train)
        train_scaled = self.scaler_.transform(train)

        self.pca_ = PCA(n_components=self.pca_variance, random_state=self.random_state).fit(train_scaled)
        self.components_ = [f"PC{i + 1}" for i in range(self.pca_.n_components_)]
        pca_train = pd.DataFrame(self.pca_.transform(train_scaled), index=train.index, columns=self.components_)
        logger.info("PCA kept %d components (%.1f%% var)", self.pca_.n_components_,
                    100 * self.pca_.explained_variance_ratio_.sum())

        # Step 1 — crisis vs typical.
        km1 = KMeans(n_clusters=2, random_state=self.random_state, n_init=20).fit(pca_train.values)
        sizes = pd.Series(km1.labels_).value_counts()
        self.crisis_label_ = int(sizes.idxmin())
        self.typical_label_ = int(sizes.idxmax())
        self.centroids_step1_ = km1.cluster_centers_

        # Step 2 — sub-regimes among the typical months (direction matters, so normalise).
        typical_mask = km1.labels_ == self.typical_label_
        typical_unit = normalize(pca_train.values[typical_mask])
        self.best_k_ = self._choose_k(typical_unit)
        km2 = KMeans(n_clusters=self.best_k_, random_state=self.random_state, n_init=20).fit(typical_unit)
        self.centroids_step2_ = km2.cluster_centers_

        # Combine: crisis -> 0, typical sub-regimes -> 1..k.
        regimes = np.empty(len(pca_train), dtype=int)
        regimes[km1.labels_ == self.crisis_label_] = 0
        regimes[typical_mask] = km2.labels_ + 1
        self.n_regimes_ = self.best_k_ + 1

        labels = pd.Series(regimes, index=train.index, name="regime")
        logger.info("Regimes found: %d (1 crisis + %d typical)", self.n_regimes_, self.best_k_)
        return labels, pca_train

    def _choose_k(self, unit_data: np.ndarray) -> int:
        """Pick k by silhouette over k_min..k_max, preferring k>=4 for regime richness."""
        best_k, best_score = 5, -np.inf
        for k in range(self.k_min, self.k_max + 1):
            labels = KMeans(n_clusters=k, random_state=self.random_state, n_init=20).fit_predict(unit_data)
            score = silhouette_score(unit_data, labels)
            logger.debug("k=%d silhouette=%.3f", k, score)
            if score > best_score and k >= 4:
                best_k, best_score = k, score
        return best_k

    # -------------------------------------------------------------- predict
    def transform_pca(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Project any panel into the fitted PCA space (train statistics only)."""
        scaled = self.scaler_.transform(panel)
        return pd.DataFrame(self.pca_.transform(scaled), index=panel.index, columns=self.components_)

    def predict(self, panel: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """Soft-classify out-of-sample months.

        Returns (hard regime = argmax, full probability frame, PCA scores).
        """
        pca = self.transform_pca(panel)
        x = pca.values

        # P(crisis), P(typical) from distance to the step-1 centroids.
        p_step1 = _softmax_from_score(euclidean_distances(x, self.centroids_step1_), self.crisis_temperature)
        p_crisis = p_step1[:, self.crisis_label_]
        p_typical = p_step1[:, self.typical_label_]

        # P(sub-regime | typical) from cosine similarity to the step-2 centroids.
        sims = cosine_similarity(normalize(x), self.centroids_step2_)
        p_sub = _softmax_from_score(sims, self.regime_temperature)

        # Law of total probability, then renormalise.
        probs = np.zeros((len(x), self.n_regimes_))
        probs[:, 0] = p_crisis
        for i in range(self.best_k_):
            probs[:, i + 1] = p_typical * p_sub[:, i]
        probs /= probs.sum(axis=1, keepdims=True)

        prob_cols = [f"prob_R{i}" for i in range(self.n_regimes_)]
        prob_df = pd.DataFrame(probs, index=panel.index, columns=prob_cols)
        hard = pd.Series(probs.argmax(axis=1), index=panel.index, name="regime")
        return hard, prob_df, pca
