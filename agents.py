import logging
from typing import Annotated, Any, Dict, TypedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from state import GraphState
from youtube_transcript_api import YouTubeTranscriptApi
import json
import os
from dotenv import load_dotenv

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables first
load_dotenv(verbose=True)
logger.debug("Current working directory: %s", os.getcwd())
logger.debug("Looking for .env file in: %s", os.path.abspath('.'))

# Check for OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    logger.debug("Available environment variables: %s", list(os.environ.keys()))
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file")
else:
    logger.debug("OPENAI_API_KEY found with length: %d", len(api_key))

def get_transcript_node(state: GraphState) -> GraphState:
    """Fetch transcript from YouTube video."""
    logger.info("Starting transcript fetch...")
    try:
        # Extract video ID from URL
        if "youtu.be" in state["video_url"]:
            video_id = state["video_url"].split("/")[-1]
        else:
            video_id = state["video_url"].split("v=")[-1].split("&")[0]
        logger.info(f"Extracted video ID: {video_id}")
        
        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        logger.info("Successfully fetched transcript")
        
        # Combine transcript segments
        full_transcript = " ".join([segment["text"] for segment in transcript])
        logger.info(f"Combined transcript length: {len(full_transcript)} characters")
        
        # Update state
        new_state = state.copy()
        new_state["transcript"] = full_transcript
        return new_state
    except Exception as e:
        logger.error(f"Error fetching transcript: {str(e)}", exc_info=True)
        new_state = state.copy()
        new_state["error"] = f"Error fetching transcript: {str(e)}"
        return new_state

def enhance_text_node(state: GraphState) -> GraphState:
    """Enhance content using RAG."""
    logger.info("Starting content enhancement...")
    try:
        if "error" in state and state["error"]:
            logger.error(f"Skipping enhancement due to previous error: {state['error']}")
            return state
            
        if "transcript" not in state or not state["transcript"]:
            error_msg = "No transcript available for enhancement"
            logger.error(error_msg)
            new_state = state.copy()
            new_state["error"] = error_msg
            return new_state
            
        # Initialize LLM
        llm = ChatOpenAI(temperature=0.7)
        logger.info("Initialized LLM")
        
        # Create vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_text(state["transcript"])
        logger.info(f"Split transcript into {len(texts)} chunks")
        
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_texts(texts, embeddings)
        logger.info("Created vector store")
        
        # Create RAG chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content enhancer. Your task is to:
            1. Analyze the provided transcript
            2. Identify key points and insights
            3. Enhance the content while maintaining accuracy
            4. Add relevant context and examples
            5. Ensure the output is professional and engaging
            6. Keep the enhanced content within 2000 characters
            
            Use the following context to enhance the content:
            {context}
            
            Original transcript:
            {transcript}"""),
            ("human", "Please enhance this content for a professional audience.")
        ])
        
        chain = prompt | llm | StrOutputParser()
        logger.info("Created RAG chain")
        
        # Get relevant context
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        context = retriever.get_relevant_documents(state["transcript"])
        context_text = "\n".join([doc.page_content for doc in context])
        logger.info(f"Retrieved {len(context)} relevant documents")
        
        # Generate enhanced content
        enhanced_text = chain.invoke({
            "context": context_text,
            "transcript": state["transcript"]
        })
        logger.info(f"Generated enhanced content of length: {len(enhanced_text)}")
        
        # Update state
        new_state = state.copy()
        new_state["enhanced_text"] = enhanced_text
        return new_state
    except Exception as e:
        logger.error(f"Error enhancing content: {str(e)}", exc_info=True)
        new_state = state.copy()
        new_state["error"] = f"Error enhancing content: {str(e)}"
        return new_state

def format_content_node(state: GraphState) -> GraphState:
    """Format content for LinkedIn."""
    logger.info("Starting content formatting...")
    try:
        if "error" in state and state["error"]:
            logger.error(f"Skipping formatting due to previous error: {state['error']}")
            return state
            
        if "enhanced_text" not in state or not state["enhanced_text"]:
            error_msg = "No enhanced content available for formatting"
            logger.error(error_msg)
            new_state = state.copy()
            new_state["error"] = error_msg
            return new_state
            
        # Initialize LLM
        llm = ChatOpenAI(temperature=0.7)
        logger.info("Initialized LLM")
        
        # Create formatting chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert LinkedIn content formatter. Your task is to:
            1. Transform the enhanced text into a professional LinkedIn post
            2. Use appropriate LinkedIn formatting (emojis, line breaks, etc.)
            3. Include relevant hashtags
            4. Add a compelling hook
            5. End with a call to action
            6. Maximum length should be around 1300 characters
            
            Enhanced text:
            {enhanced_text}"""),
            ("human", "Please format this content for LinkedIn.")
        ])
        
        chain = prompt | llm | StrOutputParser()
        logger.info("Created formatting chain")
        
        # Format content
        linkedin_formatted = chain.invoke({"enhanced_text": state["enhanced_text"]})
        logger.info(f"Generated LinkedIn post of length: {len(linkedin_formatted)}")
        
        # Update state
        new_state = state.copy()
        new_state["linkedin_formatted"] = linkedin_formatted
        logger.info("Updated state with LinkedIn formatted content")
        
        return new_state
    except Exception as e:
        logger.error(f"Error formatting content: {str(e)}", exc_info=True)
        new_state = state.copy()
        new_state["error"] = f"Error formatting content: {str(e)}"
        return new_state

def verify_content_node(state: GraphState) -> GraphState:
    """Verify the enhanced content against the original transcript."""
    logger.info("Starting content verification...")
    try:
        if "error" in state and state["error"]:
            logger.error(f"Skipping verification due to previous error: {state['error']}")
            return state
            
        if "transcript" not in state or not state["transcript"] or \
           "enhanced_text" not in state or not state["enhanced_text"]:
            error_msg = "Missing transcript or enhanced content for verification"
            logger.error(error_msg)
            new_state = state.copy()
            new_state["error"] = error_msg
            return new_state
            
        # Initialize LLM
        llm = ChatOpenAI(temperature=0)
        logger.info("Initialized LLM")
        
        # Create verification chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a content verification expert. Compare the enhanced content with the original transcript.
Check for:
1. Factual accuracy
2. Key message preservation
3. No significant deviations from original meaning
4. No added misinformation

Return a JSON with:
{{
    "verified": boolean,
    "details": {{
        "accuracy_score": float (0-1),
        "key_points_preserved": boolean,
        "deviations": [list of significant deviations],
        "recommendations": [list of improvement suggestions]
    }}
}}

Original Transcript:
{transcript}

Enhanced Content:
{enhanced_text}"""),
            ("human", "Please verify this content.")
        ])
        
        chain = prompt | llm | StrOutputParser()
        logger.info("Created verification chain")
        
        # Get verification result
        verification_result = json.loads(chain.invoke({
            "transcript": state["transcript"],
            "enhanced_text": state["enhanced_text"]
        }))
        logger.info("Successfully parsed verification result")
        
        # Update state with verification results
        new_state = state.copy()
        new_state["verification_status"] = verification_result["verified"]
        new_state["verification_details"] = verification_result["details"]
        logger.info(f"Verification status: {verification_result['verified']}")
        
        return new_state
    except Exception as e:
        logger.error(f"Error verifying content: {str(e)}", exc_info=True)
        new_state = state.copy()
        new_state["error"] = f"Error verifying content: {str(e)}"
        return new_state

def fetch_transcript_node(state: GraphState) -> GraphState:
    """Fetch transcript from YouTube video."""
    logger.info("Starting transcript fetch...")
    try:
        # Extract video ID from URL
        video_id = state["video_url"].split("v=")[-1]
        logger.info(f"Extracted video ID: {video_id}")
        
        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        logger.info("Successfully fetched transcript")
        
        # Combine transcript segments
        full_transcript = " ".join([segment["text"] for segment in transcript])
        logger.info(f"Combined transcript length: {len(full_transcript)} characters")
        
        return {**state, "transcript": full_transcript}
    except Exception as e:
        logger.error(f"Error fetching transcript: {str(e)}")
        return {**state, "error": f"Error fetching transcript: {str(e)}"}

def enhance_content_node(state: GraphState) -> GraphState:
    """Enhance content using RAG."""
    logger.info("Starting content enhancement...")
    try:
        # Initialize LLM
        llm = ChatOpenAI(temperature=0.7)
        logger.info("Initialized LLM")
        
        # Create vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_text(state["transcript"])
        logger.info(f"Split transcript into {len(texts)} chunks")
        
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_texts(texts, embeddings)
        logger.info("Created vector store")
        
        # Create RAG chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content enhancer. Your task is to:
            1. Analyze the provided transcript
            2. Identify key points and insights
            3. Enhance the content while maintaining accuracy
            4. Add relevant context and examples
            5. Ensure the output is professional and engaging
            6. Keep the enhanced content within 2000 characters
            
            Use the following context to enhance the content:
            {context}
            
            Original transcript:
            {transcript}"""),
            ("human", "Please enhance this content for a professional audience.")
        ])
        
        chain = prompt | llm | StrOutputParser()
        logger.info("Created RAG chain")
        
        # Get relevant context
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        context = retriever.get_relevant_documents(state["transcript"])
        context_text = "\n".join([doc.page_content for doc in context])
        logger.info(f"Retrieved {len(context)} relevant documents")
        
        # Generate enhanced content
        enhanced_text = chain.invoke({
            "context": context_text,
            "transcript": state["transcript"]
        })
        logger.info(f"Generated enhanced content of length: {len(enhanced_text)}")
        
        return {
            **state,
            "enhanced_text": enhanced_text,
            "vector_store": vector_store,
            "context": context_text
        }
    except Exception as e:
        logger.error(f"Error enhancing content: {str(e)}")
        return {**state, "error": f"Error enhancing content: {str(e)}"} 