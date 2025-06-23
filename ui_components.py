import streamlit as st

def render_header():
    st.set_page_config(
        page_title="Image Classifier",
        page_icon="🫡",
        layout="centered"
    )
    st.markdown("""
        <div style="text-align:center;">
            <h1 style="color:#4A90E2;">🫡Image Classification Web App</h1>
            <p style="font-size:18px;">Upload an image and let ResNet-50 identify what it sees!</p>
        </div>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color: grey;'>Built with ❤️ using Streamlit & PyTorch</div>",
        unsafe_allow_html=True
    )
