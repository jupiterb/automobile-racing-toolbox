import streamlit as st

from ui_app.src.page_layout import racing_toolbox_page_layout


def main():
    with st.expander("About the project 🏎️"):
        st.write(
            "From March 2022, the idea of the possibility of training an agent in racing simulators pushed us to create this tool. Today with Automobile Racing Toolbox you can create an agent for your favorite racing game. With the help of reinforcement learning algorithms, you don't need to collect data, just configure your environment. Unless you really want to, we can use your rides to teach the resulting agent even better."
        )
    with st.expander("How to use? ✨"):
        st.write(
            "This application is integrated with both your Trainer service and public data service. So by running any training you can use public and public resources, such as environment configurations or trip recording."
        )
        st.write("You can edit your details on the account page.")
        st.write(
            "On the following pages you can train an agent or an autoenocder. You can also train the agent from the weights and biases checkpoint."
        )
        st.write(
            "Starting each training sends a new task to the Trainer service. You can check the status of these tasks on your training tasks page and possibly cancel them."
        )


if __name__ == "__main__":
    racing_toolbox_page_layout("Automobile Training Main Page", main)
