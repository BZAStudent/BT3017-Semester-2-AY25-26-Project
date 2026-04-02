import streamlit as st

st.set_page_config(page_title="Graph Laplacian Visualiser", layout="centered")

st.markdown(
    """
    <style>
    .main-box {
        text-align: center;
        margin-top: 18vh;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-box">', unsafe_allow_html=True)
st.title("Graph Laplacian Visualiser")
st.subheader("BT3017 Project")
st.subheader("Created by Tay Jing Yao")
st.write("Visualise graph structures, spectral clustering, and range of influence propagation.")

if st.button("Start", use_container_width=False):
    st.switch_page("pages/1_GraphSelection.py")

st.markdown("</div>", unsafe_allow_html=True)