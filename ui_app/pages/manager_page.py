import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode


def main():
    st.markdown(
        f'<h1 style="color:#ee6c4d;font-size:40px;">{"Automobile Training Manager"}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("""---""")

    # TODO get list `real` trainngs
    trainings = pd.DataFrame(
        {
            "Name": ["Training 1", "Training 2"],
            "Game": ["Trackmania Nations Forever", "Forza Horizon"],
            "Algorithm": ["DQN", "BC"],
            "Status": ["Finished", "Running"],
            "Iteration": ["-", "32"],
        }
    )

    gb = GridOptionsBuilder.from_dataframe(trainings)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection(use_checkbox=True)
    gridOptions = gb.build()

    grid_response = AgGrid(
        trainings,
        gridOptions=gridOptions,
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        theme="alpine",
        width="100%",
        reload_data=True,
    )

    selected = grid_response["selected_rows"]
    try:
        row = pd.DataFrame(selected[0])
        name = str(row.iloc[0]["Name"])

        if st.button(f"Rerun {name}"):
            # TODO make training run agian
            pass

        if st.button(f"Stop {name}"):
            # TODO sop runnig
            pass
    except:
        st.write("Select one traning to rerun or stop")


if __name__ == "__main__":
    st.set_page_config(page_title="Automobile Training Manager", layout="wide")
    main()
