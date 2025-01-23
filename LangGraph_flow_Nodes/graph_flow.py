from LangGraph_flow_Nodes.All_Nodes_and_Agent import (agent, filter_node, generate, 
                                                      generate_finance_answer, check_last_tool, 
                                                      tools 
                                                      )
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition

workflow = StateGraph(MessagesState)

tools_node = ToolNode(tools)

workflow.add_node("agent", agent)
workflow.add_node("tools", tools_node)
workflow.add_node("filter", filter_node)
workflow.add_node("generate", generate)
workflow.add_node("generate_finance_answer", generate_finance_answer)

workflow.add_edge(START, "filter")

workflow.add_edge("filter", "agent")


workflow.add_conditional_edges(
    "agent",
    tools_condition,{
        "tools" : "tools",
        "__end__": END,
    },
)

workflow.add_conditional_edges(
    "tools",
    check_last_tool,
    {
        "generate": "generate",
        "generate_finance_answer": "generate_finance_answer",
        "END": END
    }
)

workflow.add_edge("generate", END)
workflow.add_edge("generate_finance_answer", END)