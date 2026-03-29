from __future__ import annotations

import random
from pathlib import Path

import networkx as nx
import streamlit as st

from graph_analysis import compute_laplacian

st.set_page_config(page_title="Select Graph", layout="centered")
st.title("Choose a Graph to Visualise")

GRAPH_OPTIONS = [
    "Tutorial 5 Question 1",
    "Tutorial 5 Question 2",
    "Tutorial 5 Question 3",
    "Custom",
    "Random",
]

choice = st.radio("Select graph type", GRAPH_OPTIONS)


# ---------- Helper functions ----------

def build_graph_from_edges(edges: list[tuple[int, int]]) -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def _distribute_nodes(node_count: int, cluster_count: int) -> list[int]:
    base_size = node_count // cluster_count
    remainder = node_count % cluster_count
    return [
        base_size + (1 if cluster_id < remainder else 0)
        for cluster_id in range(cluster_count)
    ]


def _add_sparse_component_edges(G: nx.Graph, component_nodes: list[int]) -> None:
    if len(component_nodes) <= 1:
        return

    shuffled_nodes = component_nodes[:]
    random.shuffle(shuffled_nodes)

    # Build a random spanning tree so each component is connected but not complete.
    for index in range(1, len(shuffled_nodes)):
        source = shuffled_nodes[index]
        target = random.choice(shuffled_nodes[:index])
        G.add_edge(source, target)

    possible_extra_edges = [
        (u, v)
        for i, u in enumerate(component_nodes)
        for v in component_nodes[i + 1 :]
        if not G.has_edge(u, v)
    ]

    if not possible_extra_edges:
        return

    max_extra_edges = min(len(possible_extra_edges), max(0, len(component_nodes) // 2 - 1))
    extra_edge_count = random.randint(0, max_extra_edges)
    for edge in random.sample(possible_extra_edges, extra_edge_count):
        G.add_edge(*edge)


def tutorial_example_1():
    edges = [
        (12, 2), (2, 1), (1, 5), (1, 3), (1, 4),
        (5, 11), (5, 10), (4, 8), (4, 9), (9, 7), (7, 6)
    ]
    return build_graph_from_edges(edges), 3


def tutorial_example_2():
    edges = [
        (12, 2), (5, 11), (5, 10), (5, 1), (1, 3),
        (4, 8), (4, 9), (9, 7), (7, 6)
    ]
    return build_graph_from_edges(edges), 4


def tutorial_example_3():
    edges = [
        (12, 2), (5, 11), (5, 10), (5, 1), (1, 3),
        (4, 8), (4, 9), (9, 7)
    ]
    return build_graph_from_edges(edges), 5


def random_graph(node_count: int, cluster_count: int):
    cluster_count = max(1, min(cluster_count, node_count))
    labels = list(range(1, node_count + 1))
    G = nx.Graph()
    G.add_nodes_from(labels)

    component_sizes = _distribute_nodes(node_count, cluster_count)
    start_index = 0

    for component_size in component_sizes:
        component_nodes = labels[start_index : start_index + component_size]
        _add_sparse_component_edges(G, component_nodes)
        start_index += component_size

    return G, cluster_count


def write_graph_data_file(graph: nx.Graph, cluster_count: int, graph_name: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "generated_graph_data.py"

    laplacian_result = compute_laplacian(graph)
    nodes = laplacian_result.nodes
    edges = list(graph.edges())
    adjacency_matrix = laplacian_result.adjacency.tolist()
    degree_matrix = laplacian_result.degree.tolist()
    laplacian_matrix = laplacian_result.laplacian.tolist()
    file_content = f'''from __future__ import annotations

import networkx as nx
import numpy as np

GRAPH_NAME = {graph_name!r}
CLUSTER_COUNT = {cluster_count!r}
NODES = {nodes!r}
EDGES = {edges!r}
ADJACENCY_MATRIX = np.array({adjacency_matrix!r}, dtype=float)
DEGREE_MATRIX = np.array({degree_matrix!r}, dtype=float)
LAPLACIAN_MATRIX = np.array({laplacian_matrix!r}, dtype=float)


def load_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(EDGES)
    return graph


GRAPH = load_graph()
'''
    output_path.write_text(file_content, encoding="utf-8")


def save_graph_config(graph: nx.Graph, cluster_count: int, graph_name: str):
    st.session_state["selected_graph"] = graph
    st.session_state["selected_cluster_count"] = cluster_count
    st.session_state["selected_graph_name"] = graph_name
    st.session_state["selected_nodes"] = list(graph.nodes())
    st.session_state["selected_edges"] = list(graph.edges())
    st.session_state["selected_node"] = None
    write_graph_data_file(graph, cluster_count, graph_name)


# ---------- Tutorial examples ----------
if choice == "Tutorial 5 Question 1":
    if st.button("Load Tutorial Example 1"):
        G, k = tutorial_example_1()
        save_graph_config(G, k, "Tutorial Example 1")
        st.switch_page("pages/2_Visualiser.py")

elif choice == "Tutorial 5 Question 2":
    if st.button("Load Tutorial Example 2"):
        G, k = tutorial_example_2()
        save_graph_config(G, k, "Tutorial Example 2")
        st.switch_page("pages/2_Visualiser.py")

elif choice == "Tutorial 5 Question 3":
    if st.button("Load Tutorial Example 3"):
        G, k = tutorial_example_3()
        save_graph_config(G, k, "Tutorial Example 3")
        st.switch_page("pages/2_Visualiser.py")

# ---------- Custom ----------
elif choice == "Custom":
    st.subheader("Custom Graph Setup")

    node_count = st.selectbox(
        "Number of nodes",
        options=list(range(1, 13)),
        index=5,
        key="custom_node_count",
    )

    valid_cluster_options = list(range(1, node_count + 1))

    if st.session_state.get("custom_cluster_count", 1) > node_count:
        st.session_state["custom_cluster_count"] = node_count

    cluster_count = st.selectbox(
        "Number of clusters",
        options=valid_cluster_options,
        key="custom_cluster_count",
    )

    if st.button("Create Custom Graph"):
        G, k = random_graph(node_count, cluster_count)
        save_graph_config(G, k, "Custom Graph")
        st.switch_page("pages/2_Visualiser.py")

# ---------- Random ----------
elif choice == "Random":
    st.subheader("Random Graph Setup")

    if st.button("Generate Random Graph"):
        node_count = random.randint(4, 12)
        cluster_count = random.randint(1, node_count)
        G, k = random_graph(node_count, cluster_count)
        save_graph_config(G, k, "Random Graph")
        st.switch_page("pages/2_Visualiser.py")
