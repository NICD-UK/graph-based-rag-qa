"""This is a test script for the Neo4j agent. It demonstrates how to invoke
the agent with a question in three different ways:
simple invocation, streamed response, and debug streamed response.
It is designed to be run interactively, not as a unit test.
Options are given to switch tools - with no tools the invocation methods
are near identical as there won't be intermediate messages from tool calls."""
# %% Imports
from langchain.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# lazy hack for paths. this is not a proper package!
import sys
from pathlib import Path
agent_dir = Path(__file__).parent.parent
sys.path.append(str(agent_dir))

from neo4j_agent import create_neo4j_agent, select_tools
#from tools import all_tools

# %% pick tools to use in the agent

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

# %% create agent
# use ONE of the three lines below to create the three agent versions
# (i.e. comment current line and uncomment one of the others to switch)

# agent_graph, agent_settings = create_neo4j_agent() # zero-shot no tools (shows little difference in tests as no tool calls)
# agent_graph, agent_settings = create_neo4j_agent(tools=graph_tools) # vector and graph tools case
agent_graph, agent_settings = create_neo4j_agent(tools=vector_and_calculate_tools) # vector tools case

# %% Define a question for testing 
question = """List the three departments with the highest percent support for Marine Le Pen
            during the first round of the French presidential election."""

# Define input to agent
inputs = {"messages": [{"role": "user", "content": question}]}

##################################################################
# Pick ONE innvocation method from the three cells below to test #
##################################################################

# %% 1) Simple invocation, no streaming
result = agent_graph.invoke(inputs)

result['messages'][-1].pretty_print()

# %% 2) Streamed response

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

for chunk in agent_graph.stream(inputs,
    config=config, stream_mode="values"):
    latest_message = chunk["messages"][-1]
    if latest_message.content: 
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}",
              f" with args: {[tc['args'] for tc in latest_message.tool_calls]}")

# %% 3) debugging version of stream, see everything

for chunk in agent_graph.stream(inputs, stream_mode="debug"):
    print(chunk)

# %%