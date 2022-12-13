import streamlit as st


def configure_training_resuming() -> tuple[str, str]:
    """Returns wandb run reference and checkpoint name"""
    st.write("Provide Weights and Biases run reference")
    run_reference = st.text_input("Run reference")

    st.write("Provide Weights and Biases checkpoint name")
    checkpoint_name = st.text_input("Checkpoint name")

    return run_reference, checkpoint_name
