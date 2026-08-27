"""Defines the agent used in the MoNaCo question answering pipeline
described in 'Reducing Hallucinations in Complex Question Answering using
Simple Graph-based Retrieval-Augmented Generation' by Wedge et al. (2026)
https://doi.org/10.48550/arXiv.2606.05901

Has tools to query the Neo4j database, using either vector search alone
or additional graph structure. This is based on the LangChain agent framework
with standard LangChain middleware for error handling and retrying on model errors,
and uses an Azure Chat OpenAI LLM. Tools are imported from the tools subfolder
and those specified on instantiation are exposed to the agent.

To use the agent import the create_neo4j_agent() function from this script
and call it with a list of tools to use.
"""
# %% Library imports and loading environment variables
# langchain
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call, ToolCallLimitMiddleware, ModelRetryMiddleware
from langchain.messages import ToolMessage
from langchain_openai import AzureChatOpenAI

# pydantic (for structured output definition)
from pydantic import BaseModel, Field
from typing import Optional

# import the prompts
from prompts import (GRAPH_ADDITIONAL_SUFFIX, 
                     PROMPT_PREFIX_VECTOR,
                     PROMPT_PREFIX_GRAPH,
                     PROMPT_SUFFIX,
                     N0_TOOLS_PROMPT)
from tools import all_tools

# core and ENV
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read .env file

endpoint = os.environ.get("AZURE_AI_ENDPOINT")

# %% Create llm

llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    azure_deployment=os.environ.get("AZURE_CHAT_DEPLOYMENT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    # explicitly specify reasoning for GPT-5.4
    reasoning={
        "effort": "medium", # medium was default, from low, medium, high but for 5.2 onwards none is added as default
        "summary": "auto", # auto is default
        }
)

reasoning_settings = llm.reasoning # extract the actual settings

# embeddings and neo4j connections are all within tools so no imports here

# %% define error handling middleware
@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

# %% Define a JSON response format using pydantic

# note 1 - we use same format in zero-shot cases when tool will be calculator only so references tries to reflect this
# note 2 - we use the poorly defined list not list[str] to give the model the same freedom to format the answer as the
# original monaco answers which might have nested lists or lists of dicts etc... We are not enforcing validation of what
# is in the list so it will be harder to use downstream but we only expect to pass it to the eval LLM which can hopefully
# read the answers given it needs to read the golden answers

class JSONResponse(BaseModel):
    """Structured response format for the agent's final answer."""
    answer: list = Field(description="""Answer or answers to the user's question.
                         A concise list or list of lists with one or more entities, numbers or dates only,
                         with no additional text.
                         Should be 'unknown' if the answer can't be determined from the retrieved information.""")
    explanation: str = Field(description="""Concise explanation for how the answer was determined and why it is believed to be correct,
                             including any assumptions made in generating the answer. If the answer is 'unknown',
                             the explanation should clarify why the answer could not be determined.""")
    references: Optional[list[str]] = Field(None, description="""List of article IDs used as references in the answer,
                                            if an answer was provided. Can only be null if answer is 'unknown'.
                                            If no information retrieval tools were used, this should be an empty list.""")

# %% Define function to create agent with tools

def create_neo4j_agent(tools=None, run_limit=5):
    """Create an question answering agent with the supplied tools.
    Error handling middleware is added and tool calls limited.
    If no tools are supplied an agent without tools or middleware
    is created, with a prompt reflecting lack of tools.
    Returns the graph and a settings dict to record which tools
    were available and the run limit set, so this can be stored with eval results."""
    
    if tools:
        tool_names = [tool.get_name() for tool in tools]
        settings = {"tools": tool_names,
                    "run_limit": run_limit,
                    "llm_reasoning": reasoning_settings,
                    }

        # define a prompt, import most of it but need to give the LLM the tool details
        # could do this with .format() but want variable prompts so merging strings easier
        if 'get_sections' in tool_names:   # assume graph search tools
            system_prompt = PROMPT_PREFIX_GRAPH + f"""
                The tools you have available are {tool_names}.
            """ + GRAPH_ADDITIONAL_SUFFIX + PROMPT_SUFFIX
        else: # if can't get sections it must be only vector search
            system_prompt = PROMPT_PREFIX_VECTOR + f"""
                The tools you have available are {tool_names}.
            """ + PROMPT_SUFFIX

        # create agent with tools and middleware for error handling
        graph = create_agent(
            llm,
            tools,
            middleware=[
                # Error handling
                handle_tool_errors,
                # Limit the number of tool calls to prevent infinite loops
                # run_limit is max calls per invocation; don't need thread_limit as this
                # applies max to all runs in a conversation and we won't have conversation
                ToolCallLimitMiddleware(run_limit=run_limit),
                # Retry on model errors with exponential backoff. Times in seconds
                # retry num 0 indexed so think last is retry 3 so max delay 1+4**3=65s
                ModelRetryMiddleware(max_retries=4, # 45 total attempts.
                                     backoff_factor=4, #
                                     initial_delay=1,
                                     jitter=True,
                                     max_delay=120, # allow over 60 s but max 2 min in case wrong and does 1+4**4=257s delay
                                     )
                ],
            system_prompt=system_prompt,
            response_format=JSONResponse,
        )

    else: # Agent without tools
        system_prompt = N0_TOOLS_PROMPT
        graph = create_agent(
            llm,
            middleware=[
                # Retry on model errors with exponential backoff. Times in seconds
                # retry num 0 indexed so think last is retry 3 so max delay 1+4**3=65s
                ModelRetryMiddleware(max_retries=4, # 45 total attempts.
                                     backoff_factor=4, #
                                     initial_delay=1,
                                     jitter=True,
                                     max_delay=120, # allow over 60 s but max 2 min in case wrong and does 1+4**4=257s delay
                                     )
            ], 
            system_prompt=system_prompt,
            response_format=JSONResponse,
        )
        settings = {"tools": ["No tools"],
                    "run_limit": 0, # no tools so effective 0 run limit
                    "llm_reasoning": reasoning_settings} 

    return graph, settings

# define a helper function used for selecting tools by name
def select_tools(tools_to_include: list[str] = []) -> list[object]:
    """Select a subset of tools based on their names.
    Args:
        tools_to_include (list[str]): list of tool names to include.
        Must be a subset of the names of the tools in all_tools.
    Returns:
        list[object]: list of the StructuredTools objects corresponding
        to the names in tools_to_include."""
    if type(tools_to_include) is not list:
        raise TypeError("tools_to_include must be a list of strings")
    if len(tools_to_include) < 1:
        raise ValueError("Must include at least one tool to select")
    # iterate over the list
    selected_tools = []
    missing_tools = tools_to_include.copy() # to keep track of any missing tools
    for tool in all_tools:
        if tool.name in tools_to_include:
            selected_tools.append(tool)
            missing_tools.remove(tool.name)
    # check for missing tools
    if len(missing_tools) > 0:
        raise KeyError(f"Tool(s) {missing_tools} not found in all_tools")
    return selected_tools

# %%