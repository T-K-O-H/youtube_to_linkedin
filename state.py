from typing import TypedDict, NotRequired
from langchain_community.vectorstores import Chroma

class GraphState(TypedDict):
    # Required fields
    video_url: str
    
    # Optional fields with defaults
    transcript: NotRequired[str]  # Raw transcript text
    enhanced_text: NotRequired[str]  # Enhanced content
    linkedin_formatted: NotRequired[str]  # Formatted LinkedIn post
    verification_status: NotRequired[bool]  # Verification result
    verification_details: NotRequired[dict]  # Detailed verification info
    error: NotRequired[str]  # Error message if any
    vector_store: NotRequired[Chroma]  # For RAG processing
    context: NotRequired[str]  # Context for enhancement
    formatted_content: dict 