import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import datetime

from ui_app.src.shared import Shared
from ui_app.src.utils import TaskInfo


def review_tasks():
    # trainer = Shared().trainer_service
    # tasks = trainer.get_trainings_tasks()
    tasks = [
        TaskInfo(
            task_id="582aff4e-7ed0-11ed-a491-0242ac130003",
            task_name="VAE training 0",
            task_finish_time=datetime.datetime(
                year=2022, month=12, day=17, hour=11, minute=52
            ),
            status="Done",
            result=None,
        ),
        TaskInfo(
            task_id="63ef6452-7e38-11ed-a491-0242ac130003",
            task_name="DQN training 0",
            task_finish_time=datetime.datetime(
                year=2022, month=12, day=17, hour=20, minute=13
            ),
            status="Done",
            result=None,
        ),
        TaskInfo(
            task_id="02722b0c-7dec-11ed-90d6-0242ac130003",
            task_name="continue DQN training 0",
            task_finish_time=None,
            status="Running",
            result=None,
        ),
    ]
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
        st.write("Select one traning to rerun or stop")
