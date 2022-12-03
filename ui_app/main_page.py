import streamlit as st


def main():
    st.markdown(
        f'<h1 style="color:#ee6c4d;font-size:40px;">{"Automobile Training Main Page"}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("""---""")
    with st.expander("Abour the project 🏎️"):
        st.write(
            "From March 2022, the idea of the possibility of training an agent in racing simulators pushed us to create this tool. Today with Automobile Racing Toolbox you can create an agent for your favorite racing game. With the help of reinforcement learning algorithms, you don't need to collect data, just configure your environment. Unless you really want to, we can use your rides to teach the resulting agent even better."
        )
    with st.expander("How to use? ✨"):
        st.write(
            "On the launcher page you can start a new training by selecting the game configuration, environment and the training itself. You can also use pre-trained models."
        )
        st.write(
            "On the training manager page you have a view of all your trainings. You can check their status, stop or continue them."
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Automobile Training Main", layout="wide")
    main()
