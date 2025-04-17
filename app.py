from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_ollama import ChatOllama
from langchain_huggingface import ChatHuggingFace
from tools import search_tool, guest_info_tool

chat = ChatOllama(model="llama3.1:8b-instruct-q4_0")
tools = [guest_info_tool, search_tool]
chat_with_tools = chat.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": chat_with_tools.invoke(state["messages"])
    }

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

alfred = builder.compile()

messages = [HumanMessage(content="Tell me about our guest Nikola Tesla.")]
response = alfred.invoke({"messages": messages})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)