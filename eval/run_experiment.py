"""This is the main script to run the evaluation experiment on the Monaco dataset.
Ragas loads CSVs from a datasets subfolder and stores results to an experiments subfolder.
The folder to use is specified in the 'batched_data_root_dir' variable, line 48.
Agent is instantiated with specified tools at line 226, change tools here to run different scenarios.
At line 669 within main() the experiment is run on each batch - slice the list here to limit
the number of batches for testing. The published results used 16 batches of 32, not all 1207 questions.
"""
# %% library imports
import json
from contextvars import ContextVar
from threading import Lock
import gc  # For explicit garbage collection between batches

from tqdm import tqdm

from ragas import experiment
from ragas.metrics.collections import FactualCorrectness, AnswerRelevancy
from ragas.metrics import DiscreteMetric
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from openai import AsyncOpenAI
from ragas import Dataset
import asyncio

from datetime import datetime
from random import randint

from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from langchain.messages import AIMessage

import sys
import os
from pathlib import Path

# getting environment variables (API keys)
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read .env file


# %%
# get file path to specify where to save ragas results and import agent
current_dir = Path(__file__).parent
root_dir = current_dir.parent
# root dir needed for ragas

# change this to point to the right data§
batched_data_root_dir = root_dir / 'data' / 'monaco_filtered_batched'

batched_data_root_dir = batched_data_root_dir.resolve() # ensure we have the absolute path, resolved for os
# actual dir ragas wil look in as automatically adds datasets subfolder
batched_data_dataset_dir = batched_data_root_dir / 'datasets'
batched_data_dataset_dir = batched_data_dataset_dir.resolve() # ensure we have the absolute path, resolved for os
# path for the agent code. should replace this and sort the init files instead
agent_dir = root_dir / 'langchain-agent'
agent_dir = agent_dir.resolve() # ensure we have the absolute path, resolved for os

# %%
sys.path.append(str(agent_dir)) # add the agent directory to the path so we can import the agent creation function

from neo4j_agent import create_neo4j_agent, select_tools

# define the two lists other than all that we will use later using the imported helper function
vector_and_calculate_tools = select_tools(["vector_search_paragraph", "calculate"])
graph_tools = select_tools(["articles_within_distance",
                            "calculate",
                            "get_article_text",
                            "get_backlinks",
                            "get_section_titles_and_infoboxes",
                            "get_sections",
                            "shortest_path",
                            "vector_search_article",
                            "vector_search_paragraph",
                            "window_paragraphs",
                            "window_sections",
                            ])

# %% per-turn JSONL logging — streamed as LLM/tool events fire via a LangChain callback
# One JSONL file per question, grouped under a per-run directory.
logs_dir = root_dir / "logs"
run_log_dir = logs_dir / datetime.now().strftime("%Y-%m-%d-%H%M%S")
run_log_dir.mkdir(parents=True, exist_ok=True)
print(f"Streaming per-turn JSONL logs to {run_log_dir}/ (one file per question)")


def _dump(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    return str(obj)

class JSONLTurnLogger(BaseCallbackHandler):
    """Writes one JSONL line per LLM / tool event as it happens.

    One instance per question: each ``my_ai_system`` call constructs its own
    logger bound to a unique file under ``run_log_dir``.
    """

    def __init__(self, path, qid=None, max_turns=None):
        self.path = path
        self.qid = qid
        self.max_turns = max_turns  # tool-call cap (run_limit)
        self.turn_count = 0         # LLM call count, logged but not displayed
        self.tool_count = 0         # tool-call count, displayed via tqdm.write
        self.eval_call_count = 0    # evaluator LLM calls (atomic-claim verification etc.)
        self.current_metric = None  # name of the ragas metric currently running
        self.llm_error_count = 0    # count of LLM errors (e.g., content filters)
        self._lock = Lock()

    def say(self, msg: str) -> None:
        """Print a status line above the tqdm bar and emit a 'status' JSONL event."""
        tag = f"[q{self.qid:04d}] " if isinstance(self.qid, int) else ""
        tqdm.write(f"{tag}{msg}")
        self._emit("status", message=msg)

    def _emit(self, event: str, **fields) -> None:
        record = {"event": event, "logged_at": datetime.now().isoformat(), **fields}
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    # -- LLM --------------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None,
                            tags=None, metadata=None, **kwargs):
        with self._lock:
            self.turn_count += 1
            turn_n = self.turn_count
        self._emit(
            "llm_start",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            turn=turn_n,
            max_turns=self.max_turns,
            metadata=metadata,
            tags=tags,
            model=(serialized or {}).get("id"),
            messages=[[_dump(m) for m in batch] for batch in messages],
        )

    def on_llm_end(self, response, *, run_id, parent_run_id=None, tags=None, **kwargs):
        generations = []
        for batch in getattr(response, "generations", []) or []:
            for gen in batch:
                generations.append({
                    "text": getattr(gen, "text", None),
                    "message": _dump(getattr(gen, "message", None)),
                    "generation_info": getattr(gen, "generation_info", None),
                })
        self._emit(
            "llm_end",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            llm_output=getattr(response, "llm_output", None),
            generations=generations,
        )

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        with self._lock:
            self.llm_error_count += 1
        self._emit("llm_error", run_id=str(run_id),
                   parent_run_id=str(parent_run_id) if parent_run_id else None,
                   error=repr(error))

    # -- Tools ------------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None,
                      tags=None, metadata=None, inputs=None, **kwargs):
        with self._lock:
            self.tool_count += 1
            tool_n = self.tool_count
        cap = self.max_turns if self.max_turns is not None else "?"
        self.say(f"answering question (tool {tool_n}/{cap})")
        self._emit(
            "tool_start",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tool_call_number=tool_n,
            max_tool_calls=self.max_turns,
            metadata=metadata,
            tags=tags,
            tool=(serialized or {}).get("name"),
            input=input_str,
            inputs=inputs,
        )

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self._emit(
            "tool_end",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            output=_dump(output),
        )

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._emit("tool_error", run_id=str(run_id),
                   parent_run_id=str(parent_run_id) if parent_run_id else None,
                   error=repr(error))


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

# %% create agent and wrap as ai system to generate during eval as ragas experiment requires

"""Here we create the agent and manually changed the tools to use in the agent.
Three scenarios are used:
1) no tools (comment out both tools= lines below)
2) vector, graph and calculate tools (tools=graph_tools)
3) vector and calculate tools (tools=vector_and_calculate_tools)"""

graph, agent_settings = create_neo4j_agent(
                            #tools=vector_and_calculate_tools,
                            tools=graph_tools,
                           run_limit=10, # optionally change run limit
                           ) 

# %% need a function that returns the llm response
def my_ai_system(query):
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
    thread_id = str(randint(1,10000)) # random id to avoid clashes in concurrent running
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
                # in 2025-04-01 Azure API version the token usage is in usage_metadata
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

# %% setup all eval models

# get endpoint from ENV file
endpoint = os.environ.get("AZURE_AI_ENDPOINT")

# instantiate evaluator llm using Ragas 0.4.x llm_factory
# Note this does't work in the same way as ragas 0.3.x
client = AsyncOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=f"{endpoint}/openai/v1/", # OpenAI needs this but not Azure AI embeddings
)

# pick a deployment for evaluation.
# to avoid self-recognition bias eval model (Llama) is different to generator (GPT)
evaluator_llm = llm_factory(os.environ.get("EVAL_MODEL_DEPLOYMENT"), client=client)

# %% instrument evaluator LLM so every sub-call (atomic claim, CRAG prompt, AR, …)
# shows up as a tqdm.write line under the right per-question tag.

# Per-task context so concurrent rows can't clobber each other's tag.
_current_logger: ContextVar["JSONLTurnLogger | None"] = ContextVar("_current_logger", default=None)

_orig_agenerate = evaluator_llm.agenerate

async def _logged_agenerate(prompt, response_model):
    logger = _current_logger.get()
    if logger is not None:
        with logger._lock:
            logger.eval_call_count += 1
            n = logger.eval_call_count
        metric = logger.current_metric or "judge"
        logger.say(f"  {metric}: evaluator call {n}")
        logger._emit(
            "evaluator_call_start",
            metric=metric,
            call_number=n,
            prompt_chars=len(prompt) if isinstance(prompt, str) else None,
            response_model=getattr(response_model, "__name__", str(response_model)),
        )
    return await _orig_agenerate(prompt, response_model)

evaluator_llm.agenerate = _logged_agenerate

# instantiate embeddings. 3-large is most capable of open ai models
embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-large")

# %% create the CRAG metric 

# First version used for Coarse Truthfulness. Aim to reproduce CRAG behaviour as described in
# the paper though note the exact prompt was never made public.
crag_metric = DiscreteMetric(
    name="CRAG",
    prompt="""Evaluate the LLM response: {response} against the reference answer: {reference}.
    Reference answers will be a list of correct answers with no description, just the answer.
    The LLM response may be a longer text that includes the correct answer.
    Return 1 if the response is fully correct, including all reference answers,
    0 if the response is that the LLM does not know or is unable to answer,
    and -1 if the answer is incorrect, either partially or completely.""",
    allowed_values=[-1,0,1],
)

# Additonal metric, distinguish partial correct with missing answers compared to partial correct with some incorrect
# This is what we use for the fine-grained truthfulness
# Well spotted, we call this _v3. There was a _v2 version that wasn't used in the final published evaluation so has been removed.
crag_metric_v3 = DiscreteMetric(
    name="CRAG_v3",
    prompt="""You are given a Question, a Model Prediction with Explanation for the
    prediction and an unordered list of Ground Truth answers. Judge whether the prediction matches any
    all or none of the answers from the list of Ground Truth answers.
    
    Use the question and explanation to judge whether reference and predicted answers match, rather than just string matching.
    Sometimes different words might have been used to express the same answer, for example, depending on the phrasing of the question
    False and No might be used interchangably or True and Yes. You can also allow 1% tolerance on numerical answers.
    Do not rely on your own knowledge to judge the model prediction, use the Ground Truth answers provided.

    Follow these instructions step by step to make a judgment:
    1. If the model returns 'unknown' or says that it couldn't answer the question or it doesn't have enough information
    to answer the question, then you must return 0.
    2. If the model makes a prediction, rather than saying it doesn't know, but the prediction does not match any
    of the provided answers from the Ground Truth Answer list then the prediction is wrong and you must return -1.
    3. If the model prediction matches all provided answers from the Ground Truth Answer list
    then the prediction is fully correct and you must return +1.
    4. If the model prediction matches a subset of the provided answers from the Ground Truth Answer list but some correct answers
    are missing, then the prediction is partially correct. Only if the prediction does not include any additional incorrect answers,
    then you must return +0.5.
    5. If the model prediction includes some correct answers from the Ground Truth Answer list but also includes any
    incorrect answers (answers not in the Ground Truth Answer list), then model is incorrect, and you must return -0.5.
    
    The question is {user_input}, the model prediction and explanation are {response},
    and the Ground Truth answers are {reference}.
    
    Return only one of the following values based on the instructions above: -1, +0.5, 0, -0.5 or +1.""",
    allowed_values=[-1,-0.5,0,+0.5,+1],
)

# %%
# Define the experiment
def create_experiment(batch_num): # extra factory function to allow batch num to be added to filename
    # now standard creation of experiment function with the ragas decorator
    # use name_prefix to store the date and time as prefix to the random name generated by ragas
    @experiment(name_prefix=datetime.now().strftime('%Y-%m-%d-%H%M')+f"-batch-{batch_num}")
    async def my_experiment(row):
        # Process the input through your AI system
        error_list = []  # init list to capture any gen/eval errors
        try:
            response_dict, extra_info, turn_logger = await asyncio.to_thread(my_ai_system, row["query"])
        except Exception as e:
            response_dict = None
            turn_logger = None
            extra_info = {'model': None,
                        'agent_reasoning': None,
                        'function_calls': None,
                        'token_usage': None} # default values if generation fails
            error_list.append(f"LLM generation system failed with exception: {e}")
        
        # Check if a content filter error was detected during generation
        if 'content_filter_error' in extra_info and extra_info['content_filter_error']:
            error_list.append(extra_info['content_filter_error'])
            response_dict = None  # Ensure evaluation is skipped
        
        # Check if LLM errors occurred (e.g., prompt-level content filters)
        if 'llm_error_count' in extra_info and extra_info['llm_error_count'] > 0:
            error_list.append(f"LLM encountered {extra_info['llm_error_count']} error(s) during generation (e.g., prompt-level content filter blocks)")
            response_dict = None  # Ensure evaluation is skipped

        # process the structured response
        if response_dict:
            agent_answer = response_dict.get("answer")
            explanation = response_dict.get("explanation")
            references = response_dict.get("references")
            combined_answer = f"Answer: {agent_answer}\nExplanation: {explanation}"
            if not agent_answer:
                error_list.append(f"Generation returned a structured response but answer field is empty. Returned response: {response_dict}")
                response_dict = None # treat as a failed generation so evaluation is skipped
                agent_answer = None
                explanation = None
                references = None
                combined_answer = None
        else:
            agent_answer = None
            explanation = None
            references = None
            combined_answer = None

        # instantiate standard metrics
        factual_correctness_precision = FactualCorrectness(llm=evaluator_llm,
                                                           mode="precision",
                                                           atomicity="high",
                                                           coverage="high")
        factual_correctness_recall = FactualCorrectness(llm=evaluator_llm,
                                                        mode="recall",
                                                        atomicity="high",
                                                        coverage="high")
        answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings)

        # logging
        def _log_judge(metric, inputs, result=None, err=None):
            """Append a judge/judge_error event to the per-question JSONL log."""
            if turn_logger is None:
                return
            try:
                if err is not None:
                    turn_logger._emit(
                        "judge_error",
                        metric=metric,
                        evaluator_model=evaluator_llm.model,
                        inputs=inputs,
                        error=repr(err),
                    )
                else:
                    turn_logger._emit(
                        "judge",
                        metric=metric,
                        evaluator_model=evaluator_llm.model,
                        inputs=inputs,
                        result=_dump(result),
                    )
            except Exception as log_exc:
                print(f"[turn-log] failed to log judge event for {metric}: {log_exc}")
        ctx_token = _current_logger.set(turn_logger) if turn_logger is not None else None

        # generate metric scores if a response was generated
        if response_dict: # don't attempt to evaluate if there was no response generated
            # Factual Correctness Precision
            if turn_logger is not None:
                turn_logger.current_metric = "factual_correctness_precision"
                turn_logger.say("judging response (1/6): factual_correctness_precision")
            fc_p_inputs = {"response": agent_answer, "reference": row["expected_answer"]}
            try:
                #fc_precision = await run_async_limited(
                fc_precision = await factual_correctness_precision.ascore(
                        response=str(agent_answer), # requires string not list
                        reference=row["expected_answer"]
                    )
                #)
                _log_judge("factual_correctness_precision", fc_p_inputs, result=fc_precision)
            except Exception as e:
                fc_precision = None
                _log_judge("factual_correctness_precision", fc_p_inputs, err=e)
                error_list.append(f"Factual correctness precision eval failed with exception: {e}") 
            # Factual Correctness Recall
            if turn_logger is not None:
                turn_logger.current_metric = "factual_correctness_recall"
                turn_logger.say("judging response (2/6): factual_correctness_recall")
            fc_r_inputs = {"response": agent_answer, "reference": row["expected_answer"]}
            try:
                #fc_recall = await run_async_limited(
                fc_recall = await factual_correctness_recall.ascore(
                        response=str(agent_answer), # requires string not list
                        reference=row["expected_answer"]
                    )
                #)
                _log_judge("factual_correctness_recall", fc_r_inputs, result=fc_recall)
            except Exception as e:
                fc_recall = None
                _log_judge("factual_correctness_recall", fc_r_inputs, err=e)
                error_list.append(f"Factual correctness recall eval failed with exception: {e}")
            # CRAG
            if turn_logger is not None:
                turn_logger.current_metric = "crag"
                turn_logger.say("judging response (3/6): crag")
            crag_inputs = {"response": agent_answer, "reference": row["expected_answer"]}
            try:
                crag_result = await crag_metric.ascore(
                        llm=evaluator_llm,
                        response=agent_answer,
                        reference=row["expected_answer"]
                    )
                _log_judge("crag", crag_inputs, result=crag_result)
            except Exception as e:
                crag_result = None
                _log_judge("crag", crag_inputs, err=e)
                error_list.append(f"CRAG eval failed with exception: {e}")
            # CRAG v3
            if turn_logger is not None:
                turn_logger.current_metric = "crag_v3"
                turn_logger.say("judging response (5/6): crag v3")
            crag_v3_inputs = {"user_input": row["query"], "response": combined_answer, "reference": row["expected_answer"]}
            try:
                # crag_v3_result = await run_async_limited(
                crag_v3_result = await crag_metric_v3.ascore(
                        llm=evaluator_llm,
                        user_input=row["query"],
                        response=combined_answer,
                        reference=row["expected_answer"]
                    )
                #)
                _log_judge("crag_v3", crag_v3_inputs, result=crag_v3_result)
            except Exception as e:
                crag_v3_result = None
                _log_judge("crag_v3", crag_v3_inputs, err=e)
                error_list.append(f"CRAG v3 eval failed with exception: {e}")
            # Answer Relevancy
            if turn_logger is not None:
                turn_logger.current_metric = "answer_relevancy"
                turn_logger.say("judging response (6/6): answer_relevancy")
            ar_inputs = {"user_input": row["query"], "response": combined_answer} 
            try:
                # ar_result = await run_async_limited(
                ar_result = await answer_relevancy.ascore(
                        user_input=row["query"],
                        response=combined_answer, # needs explanation
                        # does not need expected response
                    )
                #)
                _log_judge("answer_relevancy", ar_inputs, result=ar_result)
            except Exception as e:
                ar_result = None
                _log_judge("answer_relevancy", ar_inputs, err=e)
                error_list.append(f"Answer relevancy eval failed with exception: {e}")
        else:
            fc_precision = None # remember to set fc_result to None if there was an error in the generation step
            fc_recall = None # remember to set fc_result to None if there was an error in the generation step
            crag_result = None # remember to set crag_result to None if there was an error in the generation step
            crag_v3_result = None # remember to set crag_v3_result to None if there was an error in the generation step
            ar_result = None # remember to set ar_result to None if there was an error in the generation step

        if ctx_token is not None:
            _current_logger.reset(ctx_token)

        # define scores now as ragas seemsto do some form of eval causing an error
        # if this is done directly in the return step
        fc_p_score = fc_precision.value if fc_precision is not None else None
        fc_r_score = fc_recall.value if fc_recall is not None else None
        crag_score = crag_result.value if crag_result is not None else None
        crag_v3_score = crag_v3_result.value if crag_v3_result is not None else None
        ar_score = ar_result.value if ar_result is not None else None

        # Return results for metric evaluation
        return {
            **row,  # Include original data
            "agent_answer": agent_answer,  # the answer generated by the agent
            "agent_explanation": explanation, # the explanation generated by the agent
            "agent_references": references, # the references generated by the agent
            "error": error_list, # error from gen or eval step, if either failed this will be populated
            "success": len(error_list) == 0, # no error from gen or eval
            "factual_correctness_precision_score": fc_p_score, # will be value or None if eval not attempted
            "factual_correctness_recall_score": fc_r_score, # will be value or None if eval not attempted
            "crag_score": crag_score, # will be value or None if eval not attempted
            "crag_v3_score": crag_v3_score, # will be value or None if eval not attempted
            "answer_relevancy_score": ar_score, # will be value or None if eval not attempted
            "agent_settings": agent_settings, # store a dict with list of the tools and run limit
            "models": {"generator": extra_info['model'], # will be empty list if gen failed
                        "evaluator_model": evaluator_llm.model,
                        "evaluator_embeddings": embeddings.model,
                    }, 
            "agent_reasoning": extra_info['agent_reasoning'],
            "function_calls": extra_info['function_calls'],
            "token_usage": extra_info['token_usage'],
            "timestamp": datetime.now().isoformat()
        }

    return my_experiment

# %%
# little helper function to sort files by stem
def my_stem_func(e):
    return e.stem

# %% Run evaluation on the dataset
async def main():
    global THREAD_POOL
    
    # get list of paths for all files in batch
    dataset_csv_files = list(batched_data_dataset_dir.glob("*.csv"))
    dataset_csv_files.sort(key=my_stem_func) # sort by stem

    # # specify list of a single file if we have a csv with runs to repeat due to failures
    # dataset_csv_files = [batched_data_dataset_dir / "runs_to_repeat.csv"] # specify single batch for testing
    # print(f"Running repeats on file: {dataset_csv_files[0].name}")

    total_batches = len(dataset_csv_files) # count total number

    # run each batch in turn
    for file in dataset_csv_files: # slice here to limit how many batches for testing, we only used [:16] for final eval due to time/cost constraints
        batch_num = file.stem[-3:] # get the batch number from the file name
        print(f"Running experiment on batch {batch_num} of {total_batches}")
            
        # create experiment for this batch using the function defined above
        my_experiment = create_experiment(batch_num)
        # load the data for batch
        dataset = Dataset.load(name=file.stem,
                            backend="local/csv",
                            root_dir=batched_data_root_dir)
        # now run the experiment on the dataset
        results = await my_experiment.arun(dataset)
        print(f"results saved as {results.name}.csv")
        
        # Cleanup between batches to prevent memory accumulation
        del my_experiment
        del dataset
        del results
        
        # Allow pending tasks to complete and collect garbage
        await asyncio.sleep(0.1)  # Brief yield to let pending tasks settle
        gc.collect()  # Explicitly collect unreferenced objects

        print(f"Batch {batch_num} complete - memory cleaned up")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Experiment failed with exception: {e}")
    finally:
        print("All experiments complete")
