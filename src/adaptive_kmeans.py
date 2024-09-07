import numpy as np
from numpy import linalg
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_random_state


class AdaptiveKMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters=8, n_init=10, tol=1e-4, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = check_random_state(random_state)

    def fit(self, X):
        X = check_array(X)

        n, p = X.shape
        n_init = self.n_init
        n_clusters = self.n_clusters
        max_iter = self.max_iter
        tol = self.tol

        centers_opt = None
        Vt_opt = None
        partition_opt = None
        d_opt = float("inf")

        for i in range(n_init):
            # Extraction des `n_clusters` centres pris aléatoirement
            # dans `X`. On utilisera `self.random_state.choice`.
            centers_idx = ...
            centers = ...

            # Initialisation des matrices de variance--covariance
            # brutes et normalisées
            Vt = ...
            Vt_norm = ...

            step = tol + 1
            it = 0

            while step > tol and it < max_iter:
                old_centers = centers

                # Calcul d'une nouvelle partition
                dist = np.concatenate(
                    [
                        cdist(c[None, :], X, metric="mahalanobis", VI=linalg.inv(V))
                        for c, V in zip(centers, Vt_norm)
                    ]
                )
                partition = np.argmin(dist, axis=0)

                # Mise à jour des paramètres
                for k in range(n_clusters):
                    # Extraction des individus de class k
                    Xk = ...

                    # On évite les groupements dégénérés (trop peu de
                    # points pour inverser la matrice de
                    # variance--covariance empirique)
                    if Xk.shape[0] >= p:
                        # Calcul du k-ième centre
                        centers[k, :] = ...

                        # Calcul de la k-ième matrice de
                        # variance-covariance avec `np.cov`
                        c = ...

                        # Régularisation de la matrice de covariance :
                        # on grossit la diagonale pour rendre la
                        # matrice inversible quoi qu'il arrive.
                        c += 1e-5 * np.eye(c.shape[0])
                        Vt[k] = c

                        # Calcul de la matrice de variance-covariance
                        # normalisée avec `linalg.det`
                        Vt_norm[k] = ...

                step = ((old_centers - centers) ** 2).sum()
                it += 1

            # Calcul de `d_tot`. On pourra s'inspirer des instructions
            # permettant de calculer `dist` (voir plus haut).
            d_tot = ...

            # Mise à jour du modèle optimal si besoin
            if d_tot < d_opt:
                centers_opt = centers
                Vt_opt = Vt
                Vt_norm_opt = Vt_norm
                partition_opt = partition
                d_opt = d_tot

        self.labels_ = partition_opt
        self.cluster_centers_ = centers_opt
        self.covars_ = Vt_opt
        self.covars_norm_ = Vt_norm_opt
        self.d_opt = d_opt
