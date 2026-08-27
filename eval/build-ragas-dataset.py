"""Script to build a dataset suitable for passing to ragas for use with @experiment to evaluate.

Will take the questions and answer from the json and build into the csv file format expected by ragas.
A batch size is specified to limit the number of question per csv file, so that the evaluation can
be run in batches. For the published evaluation we used a batch size of 32 questions per csv file,
which gave 38 files for the 1207 questions in the filtered MoNaCo dataset. Only the first 16 files
were used for the final evaluation due to time and cost constraints.
The csv files will be saved in a datasets subdirectory of the specified path.
When Ragas runs the evaluation it will create an experiments subdirectory in the same path
to store the results, which our run_experiment.py script ensures are timestamped.
"""
# %% library imports
import json
import os
from pathlib import Path
import itertools

from ragas import Dataset

# %%
# get file path to specify where to save ragas results
current_dir = Path(__file__).parent
root_dir = current_dir.parent
data_dir = root_dir / 'data'

# %% Get the json data for building ground truth answers

monaco_path = data_dir / 'monaco_filtered.json'

with open(monaco_path, 'r') as f:
    monaco_data = json.load(f)

# %% Create a new dataset

out_dir = data_dir / 'monaco_filtered_batched'
mkdir = os.makedirs(out_dir, exist_ok=True) # create the directory if it doesn't exist

# %% Add the questions and answers to the dataset in batches

batch_size = 32 # number of questions per file
counter = 0 # initialize valueto number the files

for batch in itertools.batched(monaco_data, batch_size):
    dataset_name = f"monaco_filtered_batch_{counter:03.0f}" # zero padded batch number for sorting
    dataset = Dataset(name=dataset_name,
                  backend="local/csv",
                  root_dir=out_dir) # ragas is hard coded to create a datasets subfolder here
    for question in batch:
        dataset.append({
        "ex_num": question['ex_num'], # original question no. from MoNaCo
        "query": question['question'], # ragas required field name
        "expected_answer": question['validated_answer'],# ragas required field name
        "decomposition": question['decomposition'], # Not needed but could keep the question decomposition
        })
    dataset.save()
    out_path = out_dir / 'datasets' / (dataset.name + '.csv')
    print(f"Dataset saved to {out_path} with {len(dataset)} questions.")
    counter += 1

# %%