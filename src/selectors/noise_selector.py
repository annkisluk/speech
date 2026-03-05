import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
import pickle
from pathlib import Path


class BaseNoiseSelector(ABC):
    
    def __init__(self, feature_dim: int = 256):

        self.feature_dim = feature_dim
        self.is_fitted = False
        self.cluster_centers = {}  # {session_id: centers}
        self.num_sessions = 0
    
    @abstractmethod
    def fit_session(
        self,
        features: np.ndarray,
        session_id: int
    ) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> int:
        pass
    
    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        predictions = np.array([self.predict(f) for f in features])
        return predictions
    
    def save(self, path: str):
        """Save selector state"""
        state = {
            'feature_dim': self.feature_dim,
            'is_fitted': self.is_fitted,
            'cluster_centers': self.cluster_centers,
            'num_sessions': self.num_sessions,
            'selector_type': self.__class__.__name__
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Selector saved: {path}")
    
    def load(self, path: str):
        """Load selector state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.feature_dim = state['feature_dim']
        self.is_fitted = state['is_fitted']
        self.cluster_centers = state['cluster_centers']
        self.num_sessions = state['num_sessions']
        
        saved_type = state.get('selector_type', 'unknown')
        print(f"Selector loaded: {path} (saved as {saved_type})")


class KMeansSelector(BaseNoiseSelector):

    def __init__(
        self,
        feature_dim: int = 256,
        n_clusters: int = 20,
        random_state: int = 42
    ):

        super().__init__(feature_dim)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_models = {}  # {session_id: KMeans model}
    
    def fit_session(
        self,
        features: np.ndarray,
        session_id: int
    ) -> np.ndarray:

        print(f"Fitting K-Means for session {session_id}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Number of clusters: {self.n_clusters}")
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(-1, self.feature_dim)
        
        # Fit K-Means
        kmeans = KMeans(
            n_clusters=min(self.n_clusters, len(features)),  # Handle small datasets
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        kmeans.fit(features)
        
        # Store cluster centers
        self.cluster_centers[session_id] = kmeans.cluster_centers_
        self.kmeans_models[session_id] = kmeans
        self.num_sessions = max(self.num_sessions, session_id + 1)
        self.is_fitted = True
        
        print(f"  Fitted {len(kmeans.cluster_centers_)} clusters")
        print(f"  Inertia: {kmeans.inertia_:.2f}")
        
        return kmeans.cluster_centers_
    
    def predict(self, features: np.ndarray) -> int:
        if not self.is_fitted:
            raise ValueError("Selector not fitted. Call fit_session first.")
        
        # Ensure features are 1D
        if features.ndim > 1:
            features = features.flatten()
        
        # Calculate distance to each session's cluster centers
        min_distance = float('inf')
        best_session = 0
        
        for session_id, centers in self.cluster_centers.items():
            # Calculate L2 distance to all centers, take minimum
            distances = np.linalg.norm(centers - features, axis=1)
            min_dist_to_session = np.min(distances)
            
            if min_dist_to_session < min_distance:
                min_distance = min_dist_to_session
                best_session = session_id
        
        return best_session
    
    def get_selection_probabilities(self, features: np.ndarray) -> Dict[int, float]:

        if not self.is_fitted:
            raise ValueError("Selector not fitted")
        
        if features.ndim > 1:
            features = features.flatten()
        
        # Calculate distances to all sessions
        distances = {}
        for session_id, centers in self.cluster_centers.items():
            dists = np.linalg.norm(centers - features, axis=1)
            distances[session_id] = np.min(dists)
        
        # Convert to probabilities (inverse distance, normalized)
        inv_distances = {s: 1.0 / (d + 1e-8) for s, d in distances.items()}
        total = sum(inv_distances.values())
        probabilities = {s: v / total for s, v in inv_distances.items()}
        
        return probabilities


class MeanShiftSelector(BaseNoiseSelector):
    """Selector using MeanShift clustering.
    
    Unlike K-Means, MeanShift automatically discovers the number of clusters.
    For each session, it finds cluster centers in the feature space.
    At prediction time, it finds the nearest cluster center across all sessions.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        bandwidth: float = None,
    ):
        super().__init__(feature_dim)
        self.bandwidth = bandwidth  # None = auto-estimate per session
        self.meanshift_models = {}  # {session_id: MeanShift model}

    def fit_session(
        self,
        features: np.ndarray,
        session_id: int
    ) -> np.ndarray:
        """
        Fit MeanShift on features from one domain/session.
        Stores the discovered cluster centers for this session.
        """
        print(f"Fitting MeanShift for session {session_id}:")
        print(f"  Features shape: {features.shape}")

        if features.ndim == 1:
            features = features.reshape(-1, self.feature_dim)

        # Estimate bandwidth if not provided
        bw = self.bandwidth
        if bw is None:
            # For high-dimensional data (256-dim encoder features), sklearn's
            # estimate_bandwidth can produce values too small for bin_seeding.
            # Use median pairwise distance as a robust, data-adaptive bandwidth.
            from sklearn.metrics import pairwise_distances
            n_sub = min(500, len(features))
            idx = np.random.choice(len(features), n_sub, replace=False)
            pairwise = pairwise_distances(features[idx])
            median_dist = np.median(pairwise[pairwise > 0])
            bw = median_dist
            print(f"  Median pairwise distance: {median_dist:.4f}")
            print(f"  Using bandwidth: {bw:.4f}")

        # bin_seeding=False uses actual data points as seeds (reliable in high-dim).
        # bin_seeding=True bins the space and fails in high dimensions.
        ms = MeanShift(
            bandwidth=bw,
            bin_seeding=False,
            n_jobs=-1
        )
        ms.fit(features)

        # Store cluster centers and model
        self.cluster_centers[session_id] = ms.cluster_centers_
        self.meanshift_models[session_id] = ms
        self.num_sessions = max(self.num_sessions, session_id + 1)
        self.is_fitted = True

        n_clusters_found = len(ms.cluster_centers_)
        print(f"  Found {n_clusters_found} clusters")

        return ms.cluster_centers_

    def predict(self, features: np.ndarray) -> int:
        """Predict session ID for a single feature vector via nearest cluster center."""
        if not self.is_fitted:
            raise ValueError("Selector not fitted. Call fit_session first.")

        if features.ndim > 1:
            features = features.flatten()

        # Same logic as KMeansSelector: find nearest cluster center across all sessions
        min_distance = float('inf')
        best_session = 0

        for session_id, centers in self.cluster_centers.items():
            distances = np.linalg.norm(centers - features, axis=1)
            min_dist_to_session = np.min(distances)

            if min_dist_to_session < min_distance:
                min_distance = min_dist_to_session
                best_session = session_id

        return best_session


class GMMSelector(BaseNoiseSelector):
    def __init__(
        self,
        feature_dim: int = 256,
        n_components: int = 20,
        covariance_type: str = 'full',
        random_state: int = 42
    ):
        super().__init__(feature_dim)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm_models = {}
    
    def fit_session(
        self,
        features: np.ndarray,
        session_id: int
    ) -> np.ndarray:
        print(f"Fitting GMM for session {session_id}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Number of components: {self.n_components}")
        
        if features.ndim == 1:
            features = features.reshape(-1, self.feature_dim)
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=min(self.n_components, len(features)),
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=100
        )
        gmm.fit(features)
        
        # Store means as cluster centers
        self.cluster_centers[session_id] = gmm.means_
        self.gmm_models[session_id] = gmm
        self.num_sessions = max(self.num_sessions, session_id + 1)
        self.is_fitted = True
        
        print(f"  Fitted {len(gmm.means_)} components")
        print(f"  Converged: {gmm.converged_}")
        
        return gmm.means_
    
    def predict(self, features: np.ndarray) -> int:
        if not self.is_fitted:
            raise ValueError("Selector not fitted")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Calculate log-likelihood for each session's GMM
        max_likelihood = float('-inf')
        best_session = 0
        
        for session_id, gmm in self.gmm_models.items():
            log_likelihood = gmm.score(features)
            
            if log_likelihood > max_likelihood:
                max_likelihood = log_likelihood
                best_session = session_id
        
        return best_session
    
    def get_selection_confidence(self, features: np.ndarray) -> Dict[int, float]:
        if not self.is_fitted:
            raise ValueError("Selector not fitted")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        confidences = {}
        for session_id, gmm in self.gmm_models.items():
            confidences[session_id] = gmm.score(features)
        
        return confidences


def create_selector(
    selector_type: str = "kmeans",
    feature_dim: int = 256,
    **kwargs
) -> BaseNoiseSelector:
    if selector_type == "kmeans":
        return KMeansSelector(feature_dim=feature_dim, **kwargs)
    elif selector_type == "meanshift":
        return MeanShiftSelector(feature_dim=feature_dim, **kwargs)
    elif selector_type == "gmm":
        return GMMSelector(feature_dim=feature_dim, **kwargs)
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")

