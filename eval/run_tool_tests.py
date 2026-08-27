"""This script is not part of the main pipeline but is a minimal
test script used to look at a few different tools used in isolation,
e.g. does the calculate tool work, can we get a specified article section, etc...
Note this is designed to be run interactively and is not a unit test. It is not
for command line use - run as far as line 173 then pick a single cell after that
to run a test with different tools."""
# %% library imports
import json
import gc
from threading import Lock

from datetime import datetime
from random import randint

from langchain_core.runnables import RunnableConfig
from langchain.messages import AIMessage

import sys
from pathlib import Path

from run_experiment import JSONLTurnLogger

# getting environment variables (API keys)
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read .env file

# %%
# get file path to specify where to save ragas results and import agent
current_dir = Path(__file__).parent
root_dir = current_dir.parent
# root dir needed for ragas

# path for the agent code. should replace this and sort the __init__.py files instead
agent_dir = root_dir / 'langchain-agent'
agent_dir = agent_dir.resolve() # ensure we have the absolute path, resolved for os

# %%
sys.path.append(str(agent_dir)) # add the agent directory to the path so we can import the agent creation function

from neo4j_agent import create_neo4j_agent, select_tools

# %% per-turn JSONL logging — streamed as LLM/tool events fire via a LangChain callback
# One JSONL file per question, grouped under a per-run directory.
logs_dir = root_dir / "logs"
run_log_dir = logs_dir / datetime.now().strftime("%Y-%m-%d-%H%M%S")
run_log_dir.mkdir(parents=True, exist_ok=True)
print(f"Streaming per-turn JSONL logs to {run_log_dir}/ (one file per question)")

# JSONLTurnLogger class now imported from run_experiment

_question_counter = 0
_counter_lock = Lock()

def _new_turn_logger(thread_id: str, max_turns=None) -> JSONLTurnLogger:
    """Allocate a fresh per-question logger with a unique JSONL file path."""
    global _question_counter
    with _counter_lock:
        _question_counter += 1
        qid = _question_counter
    filename = f"q{qid:04d}-{datetime.now().strftime('%H%M%S%f')}-tid{thread_id}.jsonl"
    return JSONLTurnLogger(run_log_dir / filename, qid=qid, max_turns=max_turns)


# %% need a function that returns the llm response

# note this version takes the graph as input explicitly as want to
# test with different tools i.e. different graphs
def my_ai_system(graph, query):
    """wrapper function for agent call as we need to pass this to generate
    responses directly during the ragas experiment run
    The function takes a query as input, runs the agent, and returns the response
    along with additional information we want to capture about the generation process
    (e.g. token usage, tool calls etc) in a dict
    
    IMPORTANT:
    When the graph called by this function is changed (e.g. new tools, different prompts)
    the details should always be captured by changing the 'experiment_name' field in the
    @experiment definition
    """
    # run the agent    
    thread_id = str(randint(1,1000)) # random id to avoid clashes in concurrent running
    turn_logger = _new_turn_logger(thread_id, max_turns=agent_settings.get("run_limit"))
    # record the question once at the top of its log file
    turn_logger._emit("question", thread_id=thread_id, query=query)
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [turn_logger],
        "metadata": {"thread_id": thread_id, "query": query},
    }
    response = graph.invoke({"messages": [{"role": "user", "content": query}]},
                            config=config)
    
    # Get structured response and convert to dict
    structured_response = response.get("structured_response")
    content_filter_error = None
    
    if structured_response:
        response_dict = structured_response.model_dump()
        # dict to be split later in the eval function
    else:
        # Fallback if no structured response - check for content filter or other issues
        response_dict = None
        # Check if a content filter blocked the response
        if response["messages"]:
            last_message = response["messages"][-1]
            if isinstance(last_message, AIMessage):
                # Check for content filter indication
                response_metadata = last_message.response_metadata or {}
                incomplete_details = response_metadata.get("incomplete_details")
                if incomplete_details and incomplete_details.get("reason") == "content_filter":
                    # Extract what message was returned
                    message_text = None
                    if last_message.content:
                        if isinstance(last_message.content, str):
                            message_text = last_message.content
                        elif isinstance(last_message.content, list) and len(last_message.content) > 0:
                            if isinstance(last_message.content[0], dict):
                                message_text = last_message.content[0].get("text")
                            elif isinstance(last_message.content[0], str):
                                message_text = last_message.content[0]
                    content_filter_error = f"Content filter blocked response. Model returned: {message_text}"
                elif not last_message.content or (isinstance(last_message.content, str) and not last_message.content.strip()):
                    content_filter_error = "Generation returned empty response (possible content filter)"
    
    # collect metadata
    out_dict = {} # instantiate blank dict
    model_name = None # can't extract from final message as with structured output
                      # this is now a tool call, so wol;l collect from an AI message

    # content from final message
    # init blank dicts for details
    token_usage = []
    agent_reasoning = []
    function_calls = []
    tool_message_error = None
    # loop through messages to populate. We don't look at the Human Message
    for message in response["messages"]:
        try:
            # if isinstance(message, ToolMessage):
            #     tool_messages.append(message.content)
            if isinstance(message, AIMessage):
                # token_usage.append(message.response_metadata['token_usage'])
                token_usage.append(message.usage_metadata)
                # in 2025-04-01 API version the token usage is in usage_metadata
                # but in 2025-01-01 it was in response_metadata['token_usage']
                for item in message.content:
                    if item.get("type") == "function_call":
                        function_calls.append(item)
                    elif item.get("type") == "reasoning":
                        agent_reasoning.append(item)
                if model_name is None:
                    model_name = message.response_metadata.get('model_name')
        except Exception as e:
            tool_message_error = f"Failed extracting response details with exception: {e}"

    # add the collected data to the out dict
    out_dict['agent_reasoning'] = agent_reasoning
    out_dict['function_calls'] = function_calls
    out_dict['token_usage'] = token_usage
    out_dict['model'] = model_name
    if content_filter_error:
        out_dict['content_filter_error'] = content_filter_error  # Add content filter error to output
    if tool_message_error:
        out_dict['error'] = tool_message_error
    # Check if LLM errors occurred during generation (e.g., prompt-level content filters)
    if turn_logger and turn_logger.llm_error_count > 0:
        out_dict['llm_error_count'] = turn_logger.llm_error_count
    # also return the turn logger so the caller can append judge events to the same JSONL file
    return response_dict, out_dict, turn_logger

#########################################################################################
# Run all above to setup then pick a cell from below to run a test with different tools #
#########################################################################################

# %% create agent graph with different tools for each case

tool_test_set = select_tools(["vector_search_article",
                              ])

graph, agent_settings = create_neo4j_agent(
                            tools=tool_test_set,
                           run_limit=10, # optionally change run limit
                           ) 

resp_dict, extra, _ = my_ai_system(graph,
                                   "please get me a list of all of the sections of the wikipedia article on the 2017 german elections")

print("Response dict:")
print(json.dumps(resp_dict, indent=2))
gc.collect() # clear memory before next test

# %% create agent graph with different tools for each case

tool_test_set = select_tools(["vector_search_article",
                              "get_section_titles_and_infoboxes",])

graph, agent_settings = create_neo4j_agent(
                            tools=tool_test_set,
                           run_limit=10, # optionally change run limit
                           ) 

resp_dict, extra, _ = my_ai_system(graph,
                                   "please get me the text of all the infoboxes in the wikipedia article on Burnham, Bo. also give me all the article section titles. give your answer as a list and explain how you found the information.")

print("Response dict:")
print(json.dumps(resp_dict, indent=2))
gc.collect() # clear memory before next test

# %% create agent graph with different tools for each case

tool_test_set = select_tools(["vector_search_article",
                              "get_section_titles_and_infoboxes",
                              "get_sections"])

graph, agent_settings = create_neo4j_agent(
                            tools=tool_test_set,
                           run_limit=10, # optionally change run limit
                           ) 

resp_dict, extra, _ = my_ai_system(graph,
                                   "please get me the text of the fourth section of the wikipedia article on the 2017 german elections. your anser should be a list of the text chunks and explanation tell me how you determined which section to retrieve")


print("Response dict:")
print(json.dumps(resp_dict, indent=2))
gc.collect() # clear memory before next test

# %%

tool_test_set = select_tools(["calculate"])

graph, agent_settings = create_neo4j_agent(
                            tools=tool_test_set,
                           run_limit=10, # optionally change run limit
                           ) 

resp_dict, extra, _ = my_ai_system(graph,
                                   "please calculate the result of 3 to the power of 49 using your tool")

print("Response dict:")
print(json.dumps(resp_dict, indent=2))
print("\nExtra details:")
print(json.dumps(extra, indent=2))

# confirm the maths, remember we get a list so need first element
assert resp_dict['answer'][0] == 3**49, "The calculate tool did not return the correct answer"

gc.collect() # clear memory before next test
# %%
