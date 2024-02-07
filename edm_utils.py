import re
import pandas as pd


def parse_train_logfile(logfile_path):
    # logfile = "/n/home12/binxuwang/Github/mini_edm/exps/base_mnist_20240129-1342/std.log"
    # Define the regex pattern to extract the desired information
    # pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (\w+): (.*)"
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s+-->\s+step:\s+(\d+),\s+current lr:\s+([\d.]+)\s+average loss:\s+([\d.]+);\s+batch loss:\s+([\d.]+)"

    # Create empty lists to store the extracted information
    df_col = []
    # Read the logfile line by line and extract the desired information
    with open(logfile_path, "r") as file:
        for line in file:
            match = re.match(pattern, line)
            if match:
                timestamp = match.group(1)
                level = match.group(2)
                step = match.group(3)
                learning_rate = match.group(4)
                average_loss = match.group(5)
                batch_loss = match.group(6)
                df_col.append({"timestamp": timestamp, "level": level, "step": int(step),
                                "learning_rate": float(learning_rate), "average_loss": float(average_loss),
                                "batch_loss": float(batch_loss),})

    # Create a pandas dataframe from the extracted information
    df = pd.DataFrame(df_col)
    # Display the dataframe
    print(df.head())
    return df
