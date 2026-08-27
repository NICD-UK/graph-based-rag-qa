"""Script to merge the ragas experiment files from a batched dataset to a single file
before we analyse the results.
Also has a helper function in case any runs had to be repeated due to failures,
to safely merge repeated runs and original runs, checking for duplicates and
ensuring the agent settings and models are identical before merging."""
# %%

import pandas as pd
import os
from pathlib import Path

# %% list files

# get the experiment directory
current_dir = Path(__file__).parent
root_dir = current_dir.parent
experiment_dir = root_dir / 'data' / 'monaco_filtered_batched' / 'experiments'
out_dir = experiment_dir / 'joined_files'

# create the output directory if it doesn't exist
out_dir.mkdir(parents=True, exist_ok=True)

# list all csv files in directory

def get_files(experiment_dir: Path) -> list:
    """Get a sorted list of csv files in the experiment directory
    Assumes files are named by datetime and batch number in the format
    YYYY-MM-DD-HHMM-batch-XXX-<ragas random words>.csv so that sorting by
    name gives the correct order of batches.
    Args:
        experiment_dir (Path): directory where the csv files are located
    Returns:
        list: sorted list of csv file names
    """
    files=[]
    for entry in os.scandir(experiment_dir):
        if entry.is_file() and entry.path.endswith('.csv'):
            files.append(entry.name)
    files.sort()
    if files:
        pass
    else:
        raise FileNotFoundError(f"No files found in {experiment_dir}")
    
    return files

files = get_files(experiment_dir)

# %% Function for joining files

def join_csv_files(experiment_dir: Path, files: list) -> tuple[pd.DataFrame, str]:
    """Joins multiple csv files into a single dataframe
    Assume all files have the same columns as should be used for batches of a single experiment
    Args:
        experiment_dir (Path): directory where the csv files are located
        files (list): list of csv file names to join
    Returns:
        pd.DataFrame: joined dataframe containing all rows from the csv files
        str: name string capturing which files were used in the join
    """
    # check more than one file given
    if len(files) < 2:
        raise ValueError("At least two files are required to join")
    
    df = pd.read_csv(experiment_dir / files[0]) # read the first file to get the columns
    print(f"Joining files to: {files[0]}")
    for file in files[1:]: # loop through the remaining files and concatenate
        try:
            df_next = pd.read_csv(experiment_dir / file)
            df = pd.concat([df, df_next], ignore_index=True)
            print(f"Joined file: {file}")
        except Exception as e:
            print(f"Error joining file {file}: {e}")
    print(f"Joined {len(files)} files into a single dataframe with {len(df)} rows and {len(df.columns)} columns")
    # output a string captruing which files were used
    name_str = files[0][:26] + "to-" + files[-1][:26] + "joined"
    return df, name_str

# %% apply function for the file sets we want

# note ragas sticks everything into the experiments subdirectory of the specified path,
# so if the input dataset is the same all the outputs go in the same place
# Here we are just taking the list of data ordered files and taking the picking out the
# ones we need for the different experiments we ran and the order they were run

# all tools batch
df_all, name_str = join_csv_files(experiment_dir, files[:38])
df_all.to_csv(out_dir / f"{name_str}_all-tools.csv", index=False)
print(f"Saved joined dataframe to {out_dir / f'{name_str}_all-tools.csv'} \n")

# vector tools batch
df_vector, name_str = join_csv_files(experiment_dir, files[38:76])
df_vector.to_csv(out_dir / f"{name_str}_vector-tools.csv", index=False)
print(f"Saved joined dataframe to {out_dir / f'{name_str}_vector-tools.csv'} \n")

# zero shot batch
df_zero_shot, name_str = join_csv_files(experiment_dir, files[76:114])
df_zero_shot.to_csv(out_dir / f"{name_str}_zero-shot.csv", index=False)
print(f"Saved joined dataframe to {out_dir / f'{name_str}_zero-shot.csv'} \n")

# %%
# helper function to merge repeat experiment runs with original outputs and CHECK FOR DUPLICATES
def replace_runs_with_repeats(original_path: str,
                              repeats_path: str,
                              success_filter: bool = False) -> pd.DataFrame:
    """Replace the original runs with the repeat runs
    Filters based on 'agent_answer' being na which are the runs that were repeated
    Optonally can filter for 'success' == True but this would filter where the
    eval has failed rather than just where the generation has failed, so not the default.
    Will check for duplicates and raise a KeyError if any are found.
    Also checks to ensure agent settings and models are identical before merging.

    Args:
        original_path (str): path to the original dataframe CSV file
        repeats_path (str): path to the repeats dataframe CSV file
        success_filter (bool): whether to filter for successful runs only (default: False)
    Returns:
        pd.DataFrame: combined dataframe with original runs replaced by repeats and no duplicates
    """
    original_df = pd.read_csv(original_path)
    repeats_df = pd.read_csv(repeats_path)
    input_len = len(original_df) + len(repeats_df)
    # filter those that are na. should remove from original all rows that were included in the repeats
    original_df = original_df[original_df['agent_answer'].notna() & original_df['crag_score'].notna() & original_df['crag_v3_score'].notna()]

    repeats_df = repeats_df[repeats_df['agent_answer'].notna() & repeats_df['crag_score'].notna() & repeats_df['crag_v3_score'].notna()]

    # check settings are the same before merging
    settings_cols = ['agent_settings', 'models'] 
    for col in settings_cols:
        assert original_df[col].iloc[0] == repeats_df[col].iloc[0], f"Settings column {col} does not match between original and repeats dataframes"

    if success_filter:
        # also filter success = False
        original_df = original_df[original_df['success'] == True]
        repeats_df = repeats_df[repeats_df['success'] == True]
    # now merge
    combined_df =  pd.concat([original_df, repeats_df], ignore_index=True)
    combined_len = len(combined_df)
    # check for duplicates
    duplicates = combined_df[combined_df.duplicated(subset=['query', 'ex_num'], keep=False)]
    if not duplicates.empty:
        raise KeyError("Warning: Found duplicate rows in the combined dataframe:")
    else:
        print("Success, no duplicate rows found in the combined dataframe.")
        print(f"sum of input dataframe lengths: {input_len}")
        print(f"Combined dataframe length: {combined_len}")

    return combined_df

# %% load two dataframes
original_filename = "original_runs.csv"
repeats_filename = "repeated_generations.csv"

repeats_out_dir = experiment_dir / 'merged_files'
os.makedirs(repeats_out_dir, exist_ok=True)


combined_df = replace_runs_with_repeats(out_dir / original_filename,
                                        repeats_out_dir / repeats_filename,
                                        success_filter=True)

# save the combined dataframe
export_filename = original_filename[:-4] + "_with_reruns.csv"
combined_df.to_csv(out_dir / export_filename, index=False)
print(f"Saved combined dataframe to {out_dir / export_filename} \n")

# %%