## Evaluation code

To build a dataset in the format required by ragas run `build-ragas-dataset.py`.
This takes as input a json file of the monaco questions and produces csv question
batches according to the specified batch size.

To run the experiment including evaluation use `run_experiment.py`, having adjusted
the tools being loaded at the create agent stage for each scenario (no tools,
all tools, vector + calculate tools only). The neo4j database must be live before you
do this.

Individual tools can be tested on a single manually inputed question using
`run_tool_tests.py`. The agent can be tested using `test_neo4j_agent.py` which
is found in [../langchain-agent/utils](../langchain-agent/utils/)

We provide an alternative experiment script which limits concurrency because
ragas cannot do this so can spawn many threads leading to rate limit errors on
the LLM deployment. The `merge_experiment_batches.py` util simply joins the
output csv results files back into a single file.