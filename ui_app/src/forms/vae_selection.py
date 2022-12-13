import streamlit as st

from ui_app.src.forms.common import configure_model


def configure_vae():
    # TODO vae config source
    vae_configs = ["Trackmania-16", "Forza-32"]
    selected = st.selectbox("Select variational autencoder configuration", vae_configs)
    with st.expander("Add new variational autencoder configuration"):
        name = st.text_input(
            "Provide new variational autencoder configuration name",
            value="My_New_Vae_Config",
        )
        with st.form("new_vae_config"):
            st.write("Training parameters")
            configure_vae_training_parameters()
            st.write("Enocoder architecture")
            model = configure_model()
            submitted = st.form_submit_button("Submit")
            if submitted:
                # TODO sumbit into source
                pass


def configure_vae_training_parameters():
    lr = st.number_input(
        "Learning rate", min_value=1e-5, max_value=1.0, value=1e-4, step=5e-5
    )
    beta = st.number_input("Beta", min_value=1e-5, max_value=1.0, value=1e-4, step=5e-5)
    observation_shape = st.number_input(
        "Observtion shape", min_value=32, max_value=1028, value=128, step=32
    )
    iterations = st.number_input(
        "Iterations", min_value=10, max_value=200, step=10, value=100
    )
