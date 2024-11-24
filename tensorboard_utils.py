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
        for subdir in sorted(dirs):
            run_path = os.path.join(root, subdir)
            if os.path.isdir(run_path):
                df = extract_tensorboard_data_from_run(run_path)
                if not df.empty:
                    relative_path = os.path.relpath(run_path, root_logdir)
                    run_dataframes[relative_path] = df
                    print(f"Extracted data from {relative_path}")
    return run_dataframes


def extract_all_runs_update(root_logdir, runs_to_update=None):
    run_dataframes = {}
    for root, dirs, files in os.walk(root_logdir):
        for subdir in sorted(dirs):
            run_path = os.path.join(root, subdir)
            if os.path.isdir(run_path):
                relative_path = os.path.relpath(run_path, root_logdir)
                if runs_to_update is not None and relative_path not in runs_to_update:
                    continue
                df = extract_tensorboard_data_from_run(run_path)
                if not df.empty:
                    run_dataframes[relative_path] = df
                    print(f"Extracted data from {relative_path}")
    return run_dataframes


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


def extract_last_K_step_avg_summary(tb_data_col, simplify_runname=None, exclude_runs=(), K=1, compute_std=False):
    """
    Extracts a summary by averaging over the last k steps for each tag in TensorBoard data.

    Parameters:
    - tb_data_col (dict): Dictionary where keys are run names and values are DataFrames with 'tag', 'step', and 'value' columns.
    - simplify_runname (callable, optional): Function to simplify run names. Defaults to None.
    - exclude_runs (tuple, optional): Runs to exclude from processing. Defaults to empty tuple.
    - k (int, optional): Number of last steps to average over. Defaults to 1.

    Returns:
    - pd.DataFrame: A DataFrame containing the averaged values per tag for each run.
    """
    # Initialize a list to collect per-run DataFrames
    result_list = []
    
    # Iterate over each run in the TensorBoard data collection
    for run_name, run_data in tb_data_col.items():
        if run_name in exclude_runs:
            print(f"Excluding {run_name}")
            continue
        
        # Ensure the data is sorted by step for each tag
        run_data_sorted = run_data.sort_values(by=['tag', 'step'])
        
        # Group by 'tag' and take the last k steps for each tag
        last_k_steps = run_data_sorted.groupby('tag').tail(K)
        
        # Compute the average 'value' for each tag over the last k steps
        averaged_values = last_k_steps.groupby('tag')['value'].mean().reset_index()
        
        # Get step information: max and min step in the last k steps per run
        step_info = last_k_steps.groupby('tag')['step'].agg(['max', 'min']).reset_index()
        max_step = step_info['max'].max()
        min_step = step_info['min'].min()
        
        # Assign the run name, simplified if a function is provided
        averaged_values["run_name"] = simplify_runname(run_name) if simplify_runname is not None else run_name
        
        # Pivot the DataFrame to have tags as columns
        pivot_df = averaged_values.pivot(index='run_name', columns='tag', values='value')
        
        # Add step information
        pivot_df["full_name"] = run_name
        pivot_df["step"] = max_step
        pivot_df["step/epoch"] = min_step
        # join the std values
        if compute_std:
            std_values = last_k_steps.groupby('tag')['value'].std().reset_index()
            std_values["run_name"] = simplify_runname(run_name) if simplify_runname is not None else run_name
            pivot_df_std = std_values.pivot(index='run_name', columns='tag', values='value')
            # rename columns
            pivot_df_std.columns = [f"{col}_std" for col in pivot_df_std.columns]
            pivot_df = pivot_df.join(pivot_df_std) 

        # Append the processed DataFrame to the result list
        result_list.append(pivot_df)
    
    # Concatenate all per-run DataFrames into a single DataFrame
    if result_list:
        result_df = pd.concat(result_list)
    else:
        result_df = pd.DataFrame()
    
    return result_df