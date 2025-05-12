from langgraph.graph import StateGraph, END
from agents import get_transcript_node, enhance_text_node, format_content_node, verify_content_node
from state import GraphState

def build_graph():
    """Build the workflow graph."""
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("get_transcript", get_transcript_node)
    workflow.add_node("enhance_text", enhance_text_node)
    workflow.add_node("format_content", format_content_node)
    workflow.add_node("verify_content", verify_content_node)
    
    # Add edges
    workflow.add_edge("get_transcript", "enhance_text")
    workflow.add_edge("enhance_text", "format_content")
    workflow.add_edge("format_content", "verify_content")
    workflow.add_edge("verify_content", END)
    
    # Set entry point
    workflow.set_entry_point("get_transcript")
    
    # Compile the graph
    return workflow.compile() 