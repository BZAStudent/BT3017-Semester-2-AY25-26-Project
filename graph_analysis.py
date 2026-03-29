from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np


@dataclass
class LaplacianResult:
    nodes: List[int]
    adjacency: np.ndarray
    degree: np.ndarray
    laplacian: np.ndarray


@dataclass
class SpectralClusteringResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    zero_eigenvalue_count: int
    embedding: np.ndarray
    labels: np.ndarray
    cluster_count: int


def build_example_graph() -> nx.Graph:
    """Create a simple 10-node graph with two clear communities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(10))

    left_cluster = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4)]
    right_cluster = [(5, 6), (5, 7), (5, 8), (6, 7), (6, 9), (7, 8), (8, 9)]
    bridge_edges = [(4, 5), (2, 6)]

    graph.add_edges_from(left_cluster + right_cluster + bridge_edges)
    return graph


def compute_laplacian(graph: nx.Graph) -> LaplacianResult:
    """
    Compute adjacency matrix A, degree matrix D, and Laplacian L = D - A.
    """
    nodes = sorted(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
    degrees = adjacency.sum(axis=1)
    degree = np.diag(degrees)
    laplacian = degree - adjacency
    return LaplacianResult(nodes, adjacency, degree, laplacian)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe_norms


def _kmeans(points: np.ndarray, cluster_count: int, max_iter: int = 100) -> np.ndarray:
    """A small NumPy k-means implementation to avoid extra dependencies."""
    if cluster_count <= 1:
        return np.zeros(len(points), dtype=int)

    centers = points[:cluster_count].copy()
    labels = np.zeros(len(points), dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centers = centers.copy()
        for cluster_id in range(cluster_count):
            cluster_points = points[labels == cluster_id]
            if len(cluster_points) > 0:
                new_centers[cluster_id] = cluster_points.mean(axis=0)
        centers = new_centers

    return labels


def infer_cluster_count(eigenvalues: np.ndarray, tolerance: float = 1e-6) -> Tuple[int, int]:
    """
    Return:
    - number of eigenvalues close to zero
    - chosen cluster count for embedding / clustering
    """
    zero_count = int(np.sum(np.isclose(eigenvalues, 0.0, atol=tolerance)))

    if zero_count >= 2:
        return zero_count, zero_count

    if len(eigenvalues) <= 2:
        return zero_count, 1

    gaps = np.diff(eigenvalues[: min(len(eigenvalues), 6)])
    chosen = int(np.argmax(gaps) + 1)
    return zero_count, max(2, chosen)


def spectral_clustering(L: np.ndarray, tolerance: float = 1e-6) -> SpectralClusteringResult:
    """
    Compute eigenpairs of the Laplacian, build a spectral embedding, and assign
    nodes to clusters using row similarity in the selected eigenvectors.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    zero_count, cluster_count = infer_cluster_count(eigenvalues, tolerance=tolerance)
    embedding = eigenvectors[:, :cluster_count]
    normalized_embedding = _normalize_rows(embedding)
    labels = _kmeans(normalized_embedding, cluster_count=cluster_count)

    return SpectralClusteringResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        zero_eigenvalue_count=zero_count,
        embedding=normalized_embedding,
        labels=labels,
        cluster_count=cluster_count,
    )


def cluster_assignment_map(nodes: Sequence[int], labels: Sequence[int]) -> Dict[int, int]:
    return {int(node): int(label) for node, label in zip(nodes, labels)}


def initial_state_vector(nodes: Sequence[int], start_node: int) -> np.ndarray:
    vector = np.zeros(len(nodes))
    vector[nodes.index(start_node)] = 1.0
    return vector


def compute_influence(L: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    """
    Compute y = L^k x for a selected step k.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return x.copy()
    return np.linalg.matrix_power(L, k) @ x


def compute_influence_sequence(L: np.ndarray, x: np.ndarray, max_k: int) -> List[np.ndarray]:
    return [compute_influence(L, x, k) for k in range(max_k + 1)]


def matrix_to_string(matrix: np.ndarray, precision: int = 2) -> str:
    return np.array2string(matrix, precision=precision, suppress_small=True)
