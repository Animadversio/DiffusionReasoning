import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tensorboard_data_from_run(logdir):
    event_files = [os.path.join(logdir, f) for f in os.listdir(logdir) if 'tfevents' in f]
    
    data = []

    for event_file in event_files:
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()

        tags = event_acc.Tags()['scalars']
        for tag in tags:
            scalar_events = event_acc.Scalars(tag)
            for event in scalar_events:
                data.append({
                    'wall_time': event.wall_time,
                    'step': event.step,
                    'tag': tag,
                    'value': event.value
                })

    return pd.DataFrame(data)

def extract_all_runs(root_logdir):
    run_dataframes = {}
    for root, dirs, files in os.walk(root_logdir):
        for subdir in dirs:
            run_path = os.path.join(root, subdir)
            if os.path.isdir(run_path):
                df = extract_tensorboard_data_from_run(run_path)
                if not df.empty:
                    relative_path = os.path.relpath(run_path, root_logdir)
                    run_dataframes[relative_path] = df
                    print(f"Extracted data from {relative_path}")
    return run_dataframes


def extract_last_step_summary(tb_data_col, simplify_runname=None):
    # Create an empty dataframe
    result_df = []
    # Iterate over the runs in tb_data_col
    for run_name, run_data in tb_data_col.items():
        # Get the last step for each tag
        last_step_data = run_data.groupby('tag')['step'].max().reset_index()
        # Extract the value at the last step for each tag
        last_step_values = run_data.merge(last_step_data, on=['tag', 'step'], how='inner')
        if simplify_runname is not None:
            last_step_values["run_name"] = simplify_runname(run_name) 
        else:
            last_step_values["run_name"] = run_name
        # Transposing the data, using the tags as columns
        last_step_values = last_step_values.pivot(index='run_name', columns='tag', values='value')
        # Append the extracted values to the result dataframe
        result_df.append(last_step_values)
    result_df = pd.concat(result_df)
    return result_df


def extract_last_step_summary(tb_data_col, simplify_runname=None, exclude_runs=()):
    # Create an empty dataframe
    result_df = []
    # Iterate over the runs in tb_data_col
    for run_name, run_data in tb_data_col.items():
        if run_name in exclude_runs:
            print(f"Excluding {run_name}")
            continue
        # Get the last step for each tag
        last_step_data = run_data.groupby('tag')['step'].max().reset_index()
        maxstep = last_step_data["step"].max() #TODO, this may have issue, some experiments use epoch some use gradient step
        minstep = last_step_data["step"].min()
        # Extract the value at the last step for each tag
        last_step_values = run_data.merge(last_step_data, on=['tag', 'step'], how='inner')
        if simplify_runname is not None:
            last_step_values["run_name"] = simplify_runname(run_name) 
        else:
            last_step_values["run_name"] = run_name
        # print(last_step_values)
        # print(last_step_data)
        # Transposing the data, using the tags as columns
        last_step_values = last_step_values.pivot(index='run_name', columns='tag', values='value')
        # print(last_step_values)
        # Append the extracted values to the result dataframe
        last_step_values["full_name"] = run_name#.replace("/tensorboard_logs", "")
        last_step_values["step"] = maxstep
        last_step_values["step/epoch"] = minstep
        # raise NotImplementedError
        # last_step_values["step"] = 
        result_df.append(last_step_values)
    result_df = pd.concat(result_df)
    return result_df