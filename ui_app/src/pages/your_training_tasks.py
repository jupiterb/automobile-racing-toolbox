import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

from ui_app.src.shared import Shared
from ui_app.src.page_layout import racing_toolbox_page_layout


def main():
    trainer = Shared().trainer_service
    tasks = trainer.get_trainings_tasks()
    columns = {
        "Name": lambda task: task.task_name,
        "Id": lambda task: task.task_id,
        "Status": lambda task: task.status,
        "Finish time": lambda task: task.task_finish_time,
    }
    trainings = pd.DataFrame(
        {col: [info(task) for task in tasks] for col, info in columns.items()}
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
        id = str(row.iloc[0]["Id"])
        if st.button(f"Stop {name}"):
            try:
                trainer.stop_training(id)
            except:
                pass
    except:
        st.write("Select traning to stop it")


if __name__ == "__main__":
    racing_toolbox_page_layout("Automobile Training Tasks", main)
