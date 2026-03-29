from __future__ import annotations

import html

import numpy as np
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx

from graph_analysis import compute_laplacian, matrix_to_string


st.set_page_config(page_title="Graph Visualiser", layout="wide")
st.title("Interactive Graph Visualiser")
st.markdown(
    """
    <style>
    .graph-wrapper {
        border: 2px solid #d0d7de;
        border-radius: 14px;
        padding: 1rem;
        background: #ffffff;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
    }

    .graph-title {
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    .matrix-output {
        font-family: "Courier New", monospace;
        font-size: 0.68rem;
        line-height: 1.15;
        white-space: pre;
        overflow-x: auto;
        overflow-y: auto;
        max-height: 220px;
        padding: 0.5rem 0.65rem;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background: #f8fafc;
    }

    .eigen-output {
        font-family: "Courier New", monospace;
        font-size: 0.9rem;
        line-height: 1.3;
        white-space: pre;
        overflow-x: auto;
        overflow-y: auto;
        max-height: 260px;
        padding: 0.6rem 0.75rem;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background: #f8fafc;
    }

    .x-output {
        font-family: "Courier New", monospace;
        font-size: 1rem;
        line-height: 1.35;
        white-space: pre;
        overflow-x: auto;
        overflow-y: hidden;
        padding: 0.75rem 0.9rem;
        border: 1px solid #dbe3ee;
        border-radius: 10px;
        background: #f8fafc;
        margin-top: 0.4rem;
        margin-bottom: 1rem;
    }
    .equal-panel {
    min-height: 650px;
    display: flex;
    flex-direction: column;
    }
    .equal-panel > div {
    flex: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "selected_graph" not in st.session_state:
    st.error("No graph selected. Please go back and choose a graph first.")
    st.stop()


def render_compact_output(value: str) -> None:
    st.markdown(
        f"<div class='matrix-output'>{html.escape(value)}</div>",
        unsafe_allow_html=True,
    )


def render_eigen_output(value: str) -> None:
    st.markdown(
        f"<div class='eigen-output'>{html.escape(value)}</div>",
        unsafe_allow_html=True,
    )


def render_x_output(value: str) -> None:
    st.markdown(
        f"<div class='x-output'>{html.escape(value)}</div>",
        unsafe_allow_html=True,
    )


G = st.session_state["selected_graph"]
graph_name = st.session_state.get("selected_graph_name", "Graph")
actual_cluster_count = nx.number_connected_components(G)

laplacian_result = compute_laplacian(G)
eigenvalues, eigenvectors = np.linalg.eigh(laplacian_result.laplacian)
zero_eigenvalue_indices = np.where(np.isclose(eigenvalues, 0.0, atol=1e-6))[0]
zero_eigenvectors = eigenvectors[:, zero_eigenvalue_indices]

st.subheader(graph_name)
st.write(f"Cluster count: {actual_cluster_count}")

if "selected_node" not in st.session_state:
    st.session_state["selected_node"] = None

if "influence_k" not in st.session_state:
    st.session_state["influence_k"] = 0

selected_node = st.session_state.get("selected_node")
current_k = st.session_state.get("influence_k", 0)

actual_selected_node = None
distances = {}
max_k = 0

if selected_node is not None:
    for node in G.nodes():
        if str(node) == str(selected_node):
            actual_selected_node = node
            break

if actual_selected_node is not None:
    distances = nx.single_source_shortest_path_length(G, actual_selected_node)
    max_k = max(distances.values()) if distances else 0

    if st.session_state["influence_k"] > max_k:
        st.session_state["influence_k"] = max_k
        current_k = max_k

node_list = sorted(G.nodes())
n = len(node_list)
x_vector = np.zeros(n)

if actual_selected_node is not None:
    # initial x (k = 0)
    start_index = node_list.index(actual_selected_node)
    x_vector[start_index] = 1

    L = laplacian_result.laplacian

    # apply L repeatedly k times
    for _ in range(current_k):
        x_vector = L @ x_vector

nodes = []
for n in G.nodes():
    if actual_selected_node is None:
        node_color = "#4f8bf9"
        node_size = 22
    else:
        dist = distances.get(n, float("inf"))
        is_influenced = dist <= current_k
        node_color = "#22c55e" if is_influenced else "#ef4444"
        node_size = 30 if str(n) == str(actual_selected_node) else 22

    nodes.append(
        Node(
            id=str(n),
            label=str(n),
            size=node_size,
            color=node_color,
        )
    )

edges = []
for u, v in G.edges():
    edges.append(Edge(source=str(u), target=str(v)))

config = Config(
    width="100%",
    height=550,
    directed=False,
    physics=True,
    hierarchical=False,
    nodeHighlightBehavior=True,
    highlightColor="#F7A7A6",
    collapsible=False,
)

top_btn_col1, top_btn_col2 = st.columns(2)

with top_btn_col1:
    if st.button("Clear selected node"):
        st.session_state["selected_node"] = None
        st.session_state["influence_k"] = 0
        st.rerun()

with top_btn_col2:
    if st.button("Back to graph selection"):
        st.switch_page("pages/1_GraphSelection.py")

st.markdown(f"**Influence Row Matrix x (k = {current_k})**")
st.caption(f"Node order: {node_list}")
render_x_output(matrix_to_string(x_vector.reshape(1, -1)))

left, right = st.columns([2, 1])

with left:
    graph_box = st.container(border=True)

    with graph_box:
        st.markdown("**Graph Canvas**")

        selected = agraph(
            nodes=nodes,
            edges=edges,
            config=config,
        )

        if selected is not None and selected != st.session_state.get("selected_node"):
            st.session_state["selected_node"] = selected
            st.session_state["influence_k"] = 0
            st.rerun()

with right:
    st.subheader("Range of Influence")
    st.write("Selected node:", st.session_state.get("selected_node"))
    st.write("Cluster count:", actual_cluster_count)

    if actual_selected_node is None:
        st.info("Select a node on the graph to start the range-of-influence view.")
    else:
        st.write(f"Current k-step: {st.session_state['influence_k']}")
        st.write(f"Maximum k-step: {max_k}")

        k_col1, k_col2 = st.columns(2)

        with k_col1:
            if st.button("⬅️ Previous step", disabled=st.session_state["influence_k"] <= 0):
                st.session_state["influence_k"] -= 1
                st.rerun()

        with k_col2:
            if st.button("Next step ➡️", disabled=st.session_state["influence_k"] >= max_k):
                st.session_state["influence_k"] += 1
                st.rerun()
    
    st.subheader("Spectral Clustering")
    st.markdown("**Eigenvalues**")
    render_eigen_output(matrix_to_string(np.round(eigenvalues, 4)))

    st.markdown("**Eigenvectors for Eigenvalue 0**")
    render_eigen_output(matrix_to_string(np.round(zero_eigenvectors, 4)))

st.markdown("---")
st.subheader("Matrix Representation")

colA, colD, colL = st.columns(3)

with colA:
    st.markdown("**Adjacency Matrix (A)**")
    render_compact_output(matrix_to_string(laplacian_result.adjacency))

with colD:
    st.markdown("**Degree Matrix (D)**")
    render_compact_output(matrix_to_string(laplacian_result.degree))

with colL:
    st.markdown("**Laplacian Matrix (L)**")
    render_compact_output(matrix_to_string(laplacian_result.laplacian))