import gradio as gr
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, Annotated, List, Tuple, Union, Optional
import json
import PyPDF2
import requests
from bs4 import BeautifulSoup
import io
import trafilatura
import ragas
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset
import plotly.graph_objects as go
import numpy as np
from langchain_community.vectorstores import FAISS
import asyncio
from langchain_chroma import Chroma
from langchain.schema import Document
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragas import evaluate

# Load environment variables
load_dotenv(verbose=True)

# Verify OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API key not found.")

# Define state types
class ProcessState(TypedDict):
    video_url: str
    transcript: str
    enhanced: str
    linkedin_post: str
    verification: dict
    error: str
    status: str
    verification_score: float
    enhancement_attempts: int
    needs_improvement: bool
    research_context: str

class FineTunedModelManager:
    def __init__(self, model_name: str = "Shipmaster1/finetuned_mpnet_matryoshka_mnr"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def semantic_search(self, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Perform semantic search using the fine-tuned model."""
        # Get embeddings
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        doc_embeddings = self.model.encode(documents, convert_to_numpy=True)
        
        # Calculate similarity scores
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Get top k results
        top_results = []
        for score, doc in zip(scores, documents):
            top_results.append((doc, float(score)))
        
        # Sort by score and return top k
        return sorted(top_results, key=lambda x: x[1], reverse=True)[:top_k]
    
    def find_similar_content(self, content: str, corpus: List[str], threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find similar content in a corpus using the fine-tuned model."""
        # Get embeddings
        content_embedding = self.model.encode(content, convert_to_numpy=True)
        corpus_embeddings = self.model.encode(corpus, convert_to_numpy=True)
        
        # Calculate similarity scores
        scores = util.cos_sim(content_embedding, corpus_embeddings)[0]
        
        # Filter by threshold
        similar_content = []
        for score, text in zip(scores, corpus):
            if float(score) >= threshold:
                similar_content.append((text, float(score)))
        
        return sorted(similar_content, key=lambda x: x[1], reverse=True)
    
    def process_text(self, text: str) -> Dict[str, float]:
        """Process text using the fine-tuned model to extract features."""
        # Get embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Calculate various features
        features = {
            'embedding_norm': float(np.linalg.norm(embedding)),
            'embedding_mean': float(np.mean(embedding)),
            'embedding_std': float(np.std(embedding))
        }
        
        return features

# Initialize the model manager
model_manager = FineTunedModelManager()

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    if "youtu.be" in url:
        return url.split("/")[-1]
    return url.split("v=")[-1].split("&")[0]

def get_transcript(state: ProcessState, progress=gr.Progress()) -> ProcessState:
    """Get transcript from YouTube video."""
    try:
        progress(0.25, desc="Fetching transcript...")
        video_id = extract_video_id(state["video_url"])
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        state["transcript"] = " ".join([segment["text"] for segment in transcript])
        state["status"] = "✅ Transcript fetched"
        return state
    except Exception as e:
        state["error"] = f"⚠️ Error fetching transcript: {str(e)}"
        state["status"] = "❌ Failed to fetch transcript"
        return state

def get_chroma_collection(model_name: str = "Shipmaster1/finetuned_mpnet_matryoshka_mnr"):
    """Get or create a Chroma collection using LangChain's abstraction."""
    try:
        # Use the model manager's embeddings
        collection = Chroma(
            collection_name="youtube_videos",
            embedding_function=model_manager.embeddings,
            persist_directory="./chroma_db"
        )
        return collection
    except Exception as e:
        raise Exception(f"Error creating Chroma collection: {str(e)}")

def enhance_content(state: ProcessState, progress=gr.Progress()) -> ProcessState:
    """Enhance the transcript content with semantic search and similarity analysis."""
    try:
        if not state["transcript"]:
            return state
            
        progress(0.50, desc="Enhancing content...")
        
        # Get similar content from the vector store
        collection = get_chroma_collection()
        similar_docs = collection.similarity_search(
            state["transcript"],
            k=3
        )
        
        # Process the transcript using the fine-tuned model
        transcript_features = model_manager.process_text(state["transcript"])
        
        # Initialize LLM for content generation
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content enhancer. Transform this transcript into engaging content:

1. Identify and emphasize key points
2. Add context and examples
3. Make it more engaging and professional
4. Keep it concise (max 3000 characters)
5. Maintain factual accuracy

Transcript:
{transcript}

Similar Content for Context:
{similar_content}

Transcript Features:
{features}"""),
            ("human", "Enhance this content for a professional audience.")
        ])
        
        chain = prompt | llm | StrOutputParser()
        state["enhanced"] = chain.invoke({
            "transcript": state["transcript"],
            "similar_content": "\n".join([doc.page_content for doc in similar_docs]),
            "features": str(transcript_features)
        })
        state["status"] = "✅ Content enhanced"
        return state
    except Exception as e:
        state["error"] = f"⚠️ Error enhancing content: {str(e)}"
        state["status"] = "❌ Failed to enhance content"
        return state

def format_linkedin_post(state: ProcessState, progress=gr.Progress()) -> ProcessState:
    """Format content as a LinkedIn post."""
    try:
        if not state["enhanced"]:
            return state
            
        progress(0.75, desc="Formatting for LinkedIn...")
        
        # Initialize LLM for formatting
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Create an engaging LinkedIn post from this content. The post should be:

1. Natural and conversational - write like a real person sharing insights
2. Focused on value - emphasize practical takeaways and actionable insights
3. Authentic - avoid overused phrases or corporate speak
4. Visually clean - use line breaks and emojis sparingly and purposefully
5. Under 1500 characters

Content Preservation Rules:
- MUST maintain the exact same topic and subject matter
- MUST keep all specific examples, techniques, and exercises mentioned
- MUST preserve the original context and purpose
- MUST include all key points from the original content
- MUST maintain the same level of technical detail
- MUST keep the same target audience in mind
- MUST preserve any specific terminology or jargon that's important to the topic
- MUST maintain the same tone and expertise level

Formatting Guidelines:
- Start with a hook that grabs attention
- Share insights in a natural flow
- Use 2-3 relevant hashtags maximum
- End with a genuine call to action
- Avoid numbered lists unless absolutely necessary
- Don't use section headers or dividers
- Don't use bullet points or emoji bullets
- Don't use multiple hashtag groups

Content to transform:
{content}

Remember: The goal is to make the content more engaging while keeping ALL the original information, examples, and technical details intact."""),
            ("human", "Create a natural, engaging LinkedIn post that preserves all the original content and context.")
        ])
        
        chain = prompt | llm | StrOutputParser()
        state["linkedin_post"] = chain.invoke({"content": state["enhanced"]})
        state["status"] = "✅ LinkedIn post formatted"
        return state
    except Exception as e:
        state["error"] = f"⚠️ Error formatting LinkedIn post: {str(e)}"
        state["status"] = "❌ Failed to format LinkedIn post"
        return state

def verify_content(state: ProcessState, progress=gr.Progress()) -> ProcessState:
    """Verify the enhanced content against the original using semantic similarity."""
    try:
        if not state["enhanced"] or not state["transcript"]:
            return state
            
        progress(1.0, desc="Verifying content...")
        
        # Initialize enhancement attempts if not present
        if "enhancement_attempts" not in state:
            state["enhancement_attempts"] = 0
        
        # Calculate semantic similarity
        similar_content = model_manager.find_similar_content(
            state["enhanced"],
            [state["transcript"]],
            threshold=0.7
        )
        
        similarity_score = similar_content[0][1] if similar_content else 0.0
        
        # Initialize LLM for verification
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Verify the enhanced content against the original:

1. Check factual accuracy
2. Ensure key messages are preserved
3. Look for any misrepresentations

Return JSON in this format:
{{
    "verified": boolean,
    "score": float between 0-1,
    "feedback": string with details
}}

Original:
{original}

Enhanced:
{enhanced}

Semantic Similarity Score: {similarity_score}"""),
            ("human", "Verify this content.")
        ])
        
        chain = prompt | llm | StrOutputParser()
        verification_result = json.loads(chain.invoke({
            "original": state["transcript"],
            "enhanced": state["enhanced"],
            "similarity_score": similarity_score
        }))
        
        # Update state with verification results
        state["verification"] = verification_result
        state["verification_score"] = verification_result["score"]
        
        # Trigger agent decision if score is below threshold
        if verification_result["score"] < 0.85 and state["enhancement_attempts"] < 3:
            state["needs_improvement"] = True
            # Create improvement plan
            state = agent_decide(state)
            state["status"] = f"🔄 Planning improvements (Attempt {state['enhancement_attempts'] + 1}/3)"
        else:
            state["needs_improvement"] = False
            if verification_result["score"] >= 0.85:
                state["status"] = "✅ Content quality threshold met"
            else:
                state["status"] = "⚠️ Max enhancement attempts reached"
        
        return state
    except Exception as e:
        state["error"] = f"⚠️ Error verifying content: {str(e)}"
        state["status"] = "❌ Failed to verify content"
        return state

def should_continue(state: ProcessState) -> bool:
    """Determine if processing should continue."""
    return not state.get("error", "")

def create_workflow() -> StateGraph:
    """Create the LangGraph workflow."""
    workflow = StateGraph(ProcessState)
    
    # Add nodes
    workflow.add_node("get_transcript", get_transcript)
    workflow.add_node("enhance_content", enhance_content)
    workflow.add_node("format_linkedin", format_linkedin_post)
    workflow.add_node("verify_content", verify_content)
    workflow.add_node("agent_decide", agent_decide)
    workflow.add_node("research_content", research_content)
    workflow.add_node("enhance_again", enhance_again)
    
    # Set entry point
    workflow.set_entry_point("get_transcript")
    
    # Add edges for main flow
    workflow.add_edge("get_transcript", "enhance_content")
    workflow.add_edge("enhance_content", "format_linkedin")
    workflow.add_edge("format_linkedin", "verify_content")
    workflow.add_edge("verify_content", "agent_decide")
    
    # Add conditional edges for agentic flow
    workflow.add_conditional_edges(
        "agent_decide",
        lambda x: x["needs_improvement"],
        {
            True: "research_content",
            False: END
        }
    )
    
    # Add edges for enhancement loop
    workflow.add_edge("research_content", "enhance_again")
    workflow.add_edge("enhance_again", "verify_content")
    
    # Add conditional edges for error handling
    workflow.add_conditional_edges(
        "get_transcript",
        should_continue,
        {
            True: "enhance_content",
            False: END
        }
    )
    workflow.add_conditional_edges(
        "enhance_content",
        should_continue,
        {
            True: "format_linkedin",
            False: END
        }
    )
    workflow.add_conditional_edges(
        "format_linkedin",
        should_continue,
        {
            True: "verify_content",
            False: END
        }
    )
    workflow.add_conditional_edges(
        "verify_content",
        should_continue,
        {
            True: "agent_decide",
            False: END
        }
    )
    workflow.add_conditional_edges(
        "research_content",
        should_continue,
        {
            True: "enhance_again",
            False: END
        }
    )
    workflow.add_conditional_edges(
        "enhance_again",
        should_continue,
        {
            True: "verify_content",
            False: END
        }
    )
    
    return workflow

def process_video(video_url: str, progress=gr.Progress()) -> tuple:
    """Process YouTube video and generate LinkedIn post."""
    try:
        # Input validation
        if not video_url:
            return (
                "⚠️ Please enter a YouTube URL",  # error
                "❌ Failed: No URL provided",      # status
                "",                               # transcript
                "",                               # enhanced
                "",                               # linkedin
                ""                                # verification
            )
        
        if "youtube.com" not in video_url and "youtu.be" not in video_url:
            return (
                "⚠️ Invalid URL. Please enter a YouTube URL",  # error
                "❌ Failed: Invalid URL",                      # status
                "",                                           # transcript
                "",                                           # enhanced
                "",                                           # linkedin
                ""                                            # verification
            )
        
        # Initialize state
        initial_state = ProcessState(
            video_url=video_url,
            transcript="",
            enhanced="",
            linkedin_post="",
            verification={},
            error="",
            status="Starting..."
        )
        
        # Create and run workflow
        workflow = create_workflow()
        app = workflow.compile()
        final_state = app.invoke(initial_state)
        
        # Format verification text
        if final_state.get("verification"):
            verification_text = f"""Verification Results:
• Status: {"✅ Verified" if final_state["verification"]["verified"] else "❌ Not Verified"}
• Accuracy Score: {final_state["verification"]["score"]:.2f}
• Feedback: {final_state["verification"]["feedback"]}"""
        else:
            verification_text = ""
        
        return (
            final_state.get("error", ""),           # error
            final_state.get("status", ""),          # status
            final_state.get("transcript", ""),      # transcript
            final_state.get("enhanced", ""),        # enhanced
            final_state.get("linkedin_post", ""),   # linkedin
            verification_text                       # verification
        )
        
    except Exception as e:
        return (
            f"⚠️ Error: {str(e)}",  # error
            "❌ Processing failed",  # status
            "",                     # transcript
            "",                     # enhanced
            "",                     # linkedin
            ""                      # verification
        )

def process_from_stage(state: ProcessState, start_stage: str, progress=gr.Progress()) -> tuple:
    """Process content from a specific stage onwards."""
    try:
        # Select appropriate workflow based on stage
        if start_stage == "enhance":
            workflow = create_workflow()
            if not state["transcript"]:
                return (
                    "⚠️ No transcript available to enhance",
                    "❌ Failed: No transcript",
                    state.get("transcript", ""),
                    "",
                    "",
                    ""
                )
        elif start_stage == "format":
            workflow = create_workflow()
            if not state["enhanced"]:
                return (
                    "⚠️ No enhanced content available to format",
                    "❌ Failed: No enhanced content",
                    state.get("transcript", ""),
                    state.get("enhanced", ""),
                    "",
                    ""
                )
        else:
            workflow = create_workflow()
        
        app = workflow.compile()
        final_state = app.invoke(state)
        
        # Format verification text
        if final_state.get("verification"):
            verification_text = f"""Verification Results:
• Status: {"✅ Verified" if final_state["verification"]["verified"] else "❌ Not Verified"}
• Accuracy Score: {final_state["verification"]["score"]:.2f}
• Feedback: {final_state["verification"]["feedback"]}"""
        else:
            verification_text = ""
        
        return (
            final_state.get("error", ""),
            final_state.get("status", ""),
            final_state.get("transcript", ""),
            final_state.get("enhanced", ""),
            final_state.get("linkedin_post", ""),
            verification_text
        )
        
    except Exception as e:
        return (
            f"⚠️ Error: {str(e)}",
            "❌ Processing failed",
            state.get("transcript", ""),
            state.get("enhanced", ""),
            state.get("linkedin_post", ""),
            ""
        )

def format_verification_text(verification: dict) -> str:
    """Format verification results into a readable string."""
    if not verification:
        return ""
        
    return f"""Verification Results:
• Status: {"✅ Verified" if verification.get("verified") else "❌ Not Verified"}
• Accuracy Score: {verification.get("score", 0):.2f}
• Feedback: {verification.get("feedback", "No feedback available")}"""

def safe_json_loads(json_str: str, default: dict = None) -> dict:
    """Safely parse JSON string with error handling."""
    if default is None:
        default = {}
    try:
        return json.loads(json_str) if json_str else default
    except json.JSONDecodeError:
        return default

def format_improvement_plan(plan: dict) -> str:
    """Format the improvement plan into a readable string."""
    if not plan:
        return "No improvement plan available"
        
    text = "📋 Improvement Plan:\n\n"
    
    # Improvement Areas
    if "improvement_areas" in plan:
        text += "🎯 Priority Areas:\n"
        for area in plan["improvement_areas"]:
            text += f"• {area.get('area', 'N/A')} (Priority: {area.get('priority', 'N/A')}/5)\n"
            text += f"  Strategy: {area.get('strategy', 'N/A')}\n"
            text += f"  Research Focus: {area.get('research_focus', 'N/A')}\n\n"
    
    # Research Priorities
    if "research_priorities" in plan:
        text += "🔍 Research Priorities:\n"
        for topic in plan["research_priorities"]:
            text += f"• {topic.get('topic', 'N/A')}\n"
            text += f"  Reason: {topic.get('reason', 'N/A')}\n"
            text += f"  Expected Impact: {topic.get('expected_impact', 'N/A')}\n\n"
    
    # Enhancement Strategy
    if "enhancement_strategy" in plan:
        text += "⚡ Enhancement Strategy:\n"
        strategy = plan["enhancement_strategy"]
        text += f"• Approach: {strategy.get('approach', 'N/A')}\n"
        text += f"• Key Focus: {strategy.get('key_focus', 'N/A')}\n"
        text += "• Expected Improvements:\n"
        for imp in strategy.get("expected_improvements", []):
            text += f"  - {imp}\n"
    
    return text

def format_research_results(research: dict) -> str:
    """Format the research results into a readable string."""
    if not research:
        return "No research results available"
        
    text = "📚 Research Results:\n\n"
    
    # Focused Research
    if "focused_research" in research:
        text += "🎯 Focused Research by Area:\n"
        for area, data in research["focused_research"].items():
            text += f"• {area} (Priority: {data.get('priority', 'N/A')}/5)\n"
            text += f"  Strategy: {data.get('strategy', 'N/A')}\n"
            text += "  Key Findings:\n"
            for content in data.get("content", [])[:1]:  # Show first finding
                text += f"  - {content[:200]}...\n\n"
    
    # Additional Research
    if research.get("similar_content"):
        text += "📖 Additional Research:\n"
        for content in research["similar_content"][:2]:  # Show first two
            text += f"• {content[:200]}...\n\n"
    
    return text

def create_synthetic_dataset():
    """Generate synthetic dataset for RAG evaluation."""
    try:
        # Read synthetic data from JSON file
        with open('synthetic_data.json', 'r') as f:
            data = json.load(f)
        
        # Extract data into lists
        questions = []
        answers = []
        contexts = []
        
        for item in data['data']:
            questions.append(item['question'])
            answers.append(item['answer'])
            contexts.append(item['context'])
        
        # Create dataset
        dataset = Dataset.from_dict({
            'question': questions,
            'answer': answers,
            'context': contexts
        })
        
        return dataset
    except Exception as e:
        raise Exception(f"Error generating synthetic dataset: {str(e)}")

class SentenceTransformerWrapper:
    """Wrapper class to make SentenceTransformer compatible with RAGAS evaluation."""
    def __init__(self, model):
        self.model = model
        self.run_config = {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

    def embed_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Embed text (required by RAGAS)."""
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)

    def set_run_config(self, config: dict) -> None:
        """Set run configuration for the model."""
        self.run_config = config

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Synchronous embed function."""
        return self.embed_text(text)

def evaluate_models(dataset):
    """Evaluate embedding models using RAGAS metrics."""
    try:
        # Initialize models
        openai_model = OpenAIEmbeddings(model="text-embedding-3-small")
        
        base_mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        base_mpnet_wrapper = SentenceTransformerWrapper(base_mpnet)
        
        fine_tuned_model = SentenceTransformer("Shipmaster1/finetuned_mpnet_matryoshka_mnr")
        fine_tuned_wrapper = SentenceTransformerWrapper(fine_tuned_model)
        
        # Initialize evaluation metrics
        metrics = [
            faithfulness,          # How well answers align with context
            answer_relevancy,      # How relevant answers are to questions
            context_recall,        # How well context covers required information
            context_precision      # How focused and precise the context is
        ]
        
        # Create evaluation dataset with all required columns
        eval_dataset = Dataset.from_dict({
            "question": dataset["question"],
            "answer": dataset["answer"],
            "context": dataset["context"],
            "retrieved_contexts": [[ctx] for ctx in dataset["context"]],  # Each context in its own list
            "reference": dataset["context"]  # Using context as reference for recall calculation
        })
        
        # Evaluate each model and store results
        results = {}
        
        # OpenAI model evaluation
        openai_eval = evaluate(
            eval_dataset,
            metrics=metrics,
            embeddings=openai_model,
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        )
        results["OpenAI"] = {
            "faithfulness": float(openai_eval._repr_dict["faithfulness"]),
            "answer_relevancy": float(openai_eval._repr_dict["answer_relevancy"]),
            "context_recall": float(openai_eval._repr_dict["context_recall"]),
            "context_precision": float(openai_eval._repr_dict["context_precision"])
        }
        
        # Base MPNet evaluation
        base_mpnet_eval = evaluate(
            eval_dataset,
            metrics=metrics,
            embeddings=base_mpnet_wrapper,
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        )
        results["Base MPNet"] = {
            "faithfulness": float(base_mpnet_eval._repr_dict["faithfulness"]),
            "answer_relevancy": float(base_mpnet_eval._repr_dict["answer_relevancy"]),
            "context_recall": float(base_mpnet_eval._repr_dict["context_recall"]),
            "context_precision": float(base_mpnet_eval._repr_dict["context_precision"])
        }
        
        # Fine-tuned MPNet evaluation
        fine_tuned_eval = evaluate(
            eval_dataset,
            metrics=metrics,
            embeddings=fine_tuned_wrapper,
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        )
        results["Fine-tuned MPNet"] = {
            "faithfulness": float(fine_tuned_eval._repr_dict["faithfulness"]),
            "answer_relevancy": float(fine_tuned_eval._repr_dict["answer_relevancy"]),
            "context_recall": float(fine_tuned_eval._repr_dict["context_recall"]),
            "context_precision": float(fine_tuned_eval._repr_dict["context_precision"])
        }
        
        return results
        
    except Exception as e:
        print(f"Error evaluating models: {str(e)}")
        return {}

def create_comparison_plot(results):
    """Create a comparison plot of the evaluation metrics."""
    # Define metrics we're using
    metrics = [
        'faithfulness',
        'answer_relevancy',
        'context_recall',
        'context_precision'
    ]
    
    # Extract scores for each model
    models = list(results.keys())
    model_scores = {
        model: [results[model][metric] for metric in metrics]
        for model in models
    }
    
    fig = go.Figure()
    
    # Add traces for each model
    colors = {
        "OpenAI": 'rgb(55, 83, 109)',
        "Base MPNet": 'rgb(26, 118, 255)',
        "Fine-tuned MPNet": 'rgb(15, 196, 141)'
    }
    
    for model in models:
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=model_scores[model],
            marker_color=colors.get(model, 'rgb(128, 128, 128)')
        ))
    
    # Update layout
    fig.update_layout(
        title='Model Comparison Metrics',
        xaxis_title='Metrics',
        yaxis_title='Score',
        barmode='group',
        yaxis=dict(range=[0, 1]),
        showlegend=True
    )
    
    return fig

def run_ragas_evaluation():
    """Run the complete RAGAS evaluation process."""
    try:
        # Generate synthetic dataset
        dataset = create_synthetic_dataset()
        
        # Evaluate models
        results = evaluate_models(dataset)
        
        # Create comparison plot
        plot = create_comparison_plot(results)
        
        # Format results as markdown
        results_md = """## Model Evaluation Results

### Models Being Compared
- **OpenAI Model**: text-embedding-3-small
- **Base MPNet**: sentence-transformers/all-mpnet-base-v2
- **Fine-tuned Model**: Shipmaster1/finetuned_mpnet_matryoshka_mnr

### OpenAI Model (text-embedding-3-small)
"""
        for metric in results['OpenAI']:
            results_md += f"- {metric}: {results['OpenAI'][metric]:.3f}\n"
        
        results_md += "\n### Base MPNet Model (all-mpnet-base-v2)\n"
        for metric in results['Base MPNet']:
            results_md += f"- {metric}: {results['Base MPNet'][metric]:.3f}\n"
        
        results_md += "\n### Fine-tuned Model (finetuned_mpnet_matryoshka_mnr)\n"
        for metric in results['Fine-tuned MPNet']:
            results_md += f"- {metric}: {results['Fine-tuned MPNet'][metric]:.3f}\n"
        
        return results_md, plot
    except Exception as e:
        return f"Error during evaluation: {str(e)}", None

def create_ui():
    with gr.Blocks(theme='JohnSmith9982/small_and_pretty') as demo:
        current_state = gr.State({
            "video_url": "",
            "transcript": "",
            "enhanced": "",
            "linkedin_post": "",
            "verification": {},
            "error": "",
            "status": "",
            "improvement_plan": {},
            "research_context": "{}",
            "enhancement_attempts": 0,
            "needs_improvement": False
        })
        
        gr.Markdown(
            """
            # YouTube to LinkedIn Post Converter
            Transform your YouTube videos into professional LinkedIn posts with AI content enhancement.
            
            ### 🎬 Sample Videos to Try
            Copy any of these URLs to test the application:
            ```
            1. Open AI video: https://www.youtube.com/watch?v=LsMxX86mm2Y
               Agent will likely find high quality initial content and not improve
            
            2. Financial News: https://www.youtube.com/watch?v=hvP1UNALZ3g
               Agent will likely decide to not improve this post
            
            3. Video About AI: https://www.youtube.com/watch?v=Yq0QkCxoTHM
               Agent will likely decide to improve this post
            ```
            These videos are chosen to show the application's ability to handle different types of professional content.
            """
        )
        
        with gr.Row():
            with gr.Column():
                video_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=e1GJ5tZePjk",
                    show_label=True
                )
                youtube_convert_btn = gr.Button("🚀 Generate from YouTube", variant="primary", size="lg")

        status = gr.Textbox(
            label="Status",
            value="Ready to process...",
            interactive=False
        )
        
        error = gr.Textbox(
            label="Error",
            visible=False,
            interactive=False
        )

        with gr.Tabs() as tabs:
            with gr.TabItem("📝 Content"):
                with gr.Row():
                    with gr.Column():
                        transcript = gr.TextArea(
                            label="📄 Raw Transcript",
                            interactive=False,
                            show_copy_button=True,
                            lines=8
                        )
                    with gr.Column():
                        enhanced = gr.TextArea(
                            label="✨ Enhanced Content",
                            interactive=False,
                            show_copy_button=True,
                            lines=8
                        )
                
                with gr.Row():
                    with gr.Column():
                        linkedin = gr.TextArea(
                            label="🔗 LinkedIn Post",
                            interactive=False,
                            show_copy_button=True,
                            lines=6
                        )
                
                with gr.Row():
                    with gr.Column():
                        verification = gr.TextArea(
                            label="✓ Verification Results",
                            interactive=False,
                            lines=4
                        )
                
                with gr.Row():
                    with gr.Column():
                        improvement_plan = gr.TextArea(
                            label="📋 Improvement Plan",
                            interactive=False,
                            show_copy_button=True,
                            lines=8,
                            visible=True,
                            value="Waiting for verification..."
                        )
                
                with gr.Row():
                    with gr.Column():
                        research_results = gr.TextArea(
                            label="🔍 Research Results",
                            interactive=False,
                            show_copy_button=True,
                            lines=8,
                            visible=True,
                            value="Waiting for research..."
                        )
                
                with gr.Row():
                    with gr.Column():
                        improved_linkedin = gr.TextArea(
                            label="🚀 Improved LinkedIn Post Final",
                            interactive=False,
                            show_copy_button=True,
                            lines=6,
                            visible=True,
                            value="Waiting for improvements..."
                        )

                # Loading indicators
                with gr.Row(visible=False) as loading_indicators:
                    transcript_loading = gr.Markdown("🔄 Fetching transcript...")
                    enhanced_loading = gr.Markdown("🔄 Enhancing content...")
                    linkedin_loading = gr.Markdown("🔄 Formatting for LinkedIn...")
                    verify_loading = gr.Markdown("🔄 Verifying content...")
                    plan_loading = gr.Markdown("🔄 Creating improvement plan...")
                    research_loading = gr.Markdown("🔄 Researching content...")
                    improved_loading = gr.Markdown("🔄 Creating improved post...")
            
            with gr.TabItem("ℹ️ Help"):
                gr.Markdown(
                    """
                    ### How to Use
                    1. **Input**: Paste a YouTube video URL in the input field
                    2. **Process**: Click the "Generate Post" button
                    3. **Wait**: The system will process your video through multiple steps
                    4. **Review**: Check the generated content in each tab
                    5. **Copy**: Use the copy button to grab your LinkedIn post
                    
                    ### 🔄 Regeneration Options
                    - Click 🔄 next to "Enhanced Content" to regenerate from the enhancement stage
                    - Click 🔄 next to "LinkedIn Post" to regenerate from the formatting stage
                    
                    ### 💡 Tips for Best Results
                    - Use videos with clear English audio
                    - Optimal video length: 5-15 minutes
                    - Ensure videos have accurate captions
                    - Review and personalize the post before sharing
                    - Consider your target audience when selecting videos
                    
                    """
                )

            with gr.TabItem("RAGAS Evaluation"):
                gr.Markdown(
                    """
                    # RAGAS Model Evaluation
                    Compare the performance of three embedding models using synthetic data.
                    
                    ### Models Being Evaluated
                    - **OpenAI Model**: text-embedding-3-small (Not Free)
                    - **Base MPNet**: sentence-transformers/all-mpnet-base-v2 (Open Source)
                    - **Fine-tuned Model**: Shipmaster1/finetuned_mpnet_matryoshka_mnr (Free Custom, trained on YouTube transcript handling)
                    
                    The evaluation uses GPT-3.5 Turbo to assess the quality of the embeddings through various metrics:
                    - Faithfulness: How well the answers align with the provided context
                    - Answer Relevancy: How relevant the answers are to the questions
                    - Context Recall: How well the model retrieves relevant context
                    - Context Precision: How precise the retrieved context is
                    
                    
                    Click the run button to find out how well the models perform on the synthetic data.
                    """
                )
                
                with gr.Row():
                    evaluate_btn = gr.Button("Run Evaluation", variant="primary", size="lg")
                
                with gr.Row():
                    results_md = gr.Markdown(label="Evaluation Results")
                    plot_output = gr.Plot(label="Comparison Plot")
                
                evaluate_btn.click(
                    fn=run_ragas_evaluation,
                    outputs=[results_md, plot_output]
                )

        def update_loading_state(stage: str):
            """Update loading indicators based on current stage."""
            states = {
                "transcript": [True, False, False, False, False, False, False],
                "enhance": [False, True, False, False, False, False, False],
                "format": [False, False, True, False, False, False, False],
                "verify": [False, False, False, True, False, False, False],
                "plan": [False, False, False, False, True, False, False],
                "research": [False, False, False, False, False, True, False],
                "improved": [False, False, False, False, False, False, True],
                "done": [False, False, False, False, False, False, False]
            }
            
            # Loading messages for each stage
            loading_messages = {
                "transcript": "🔄 Fetching transcript...\n⏳ Please wait...",
                "enhance": "✨ Enhancing content...\n⚡ AI is working its magic...",
                "format": "🎨 Formatting for LinkedIn...\n📝 Creating engaging post...",
                "verify": "🔍 Verifying content...\n⚖️ Checking accuracy...",
                "plan": "🔄 Creating improvement plan...",
                "research": "🔎 Researching content...\n📚 Finding relevant information...",
                "improved": "🚀 Creating improved LinkedIn post...\n✨ Applying enhancements..."
            }
            
            # Get current stage message
            current_message = loading_messages.get(stage, "")
            
            # Return loading states and message
            return [
                gr.update(visible=state) for state in states.get(stage, [False] * 7)
            ], current_message

        def process_with_loading(url, state):
            """Process video with loading indicators."""
            try:
                # Initialize state if needed
                if "improvement_plan" not in state:
                    state["improvement_plan"] = {}
                if "research_context" not in state:
                    state["research_context"] = "{}"
                if "enhancement_attempts" not in state:
                    state["enhancement_attempts"] = 0
                if "needs_improvement" not in state:
                    state["needs_improvement"] = False
                    
                # Show loading indicators
                loading_states, message = update_loading_state("transcript")
                yield [
                    "",  # error
                    "Processing...",  # status
                    message,  # transcript (loading)
                    "",  # enhanced
                    "",  # linkedin
                    "",  # verification
                    "Waiting for verification...",  # improvement plan
                    "Waiting for research...",  # research results
                    "Waiting for improvements...",  # improved linkedin
                    state,  # current_state
                    *loading_states  # loading indicators
                ]
                
                # Get transcript
                state["video_url"] = url
                transcript_text = get_transcript(state)["transcript"]
                
                # Show enhancing state
                loading_states, message = update_loading_state("enhance")
                yield [
                    "",
                    "Enhancing content...",
                    transcript_text,
                    message,  # enhanced (loading)
                    "",
                    "",
                    "",
                    "",
                    "",
                    state,
                    *loading_states
                ]
                
                # Enhance content
                state["transcript"] = transcript_text
                enhanced_state = enhance_content(state)
                enhanced_text = enhanced_state["enhanced"]
                
                # Show formatting state
                loading_states, message = update_loading_state("format")
                yield [
                    "",
                    "Formatting for LinkedIn...",
                    transcript_text,
                    enhanced_text,
                    message,  # linkedin (loading)
                    "",
                    "",
                    "",
                    "",
                    state,
                    *loading_states
                ]
                
                # Format LinkedIn post
                state["enhanced"] = enhanced_text
                linkedin_state = format_linkedin_post(state)
                linkedin_text = linkedin_state["linkedin_post"]
                
                # Show verifying state
                loading_states, message = update_loading_state("verify")
                yield [
                    "",
                    "Verifying content...",
                    transcript_text,
                    enhanced_text,
                    linkedin_text,
                    "🔍 Verifying...\n⚖️ Analyzing accuracy...",  # verification (loading)
                    "",
                    "",
                    "",
                    state,
                    *loading_states
                ]
                
                # Verify content
                state["linkedin_post"] = linkedin_text
                final_state = verify_content(state)
                verification_text = format_verification_text(final_state.get("verification", {}))
                
                # Update improvement plan and research results
                improvement_plan_text = format_improvement_plan(final_state.get("improvement_plan", {}))
                research_results_text = format_research_results(safe_json_loads(final_state.get("research_context", "{}")))
                
                # Check if enhancement is needed
                if final_state.get("needs_improvement", False):
                    # Show planning state
                    loading_states, message = update_loading_state("plan")
                    yield [
                        "",
                        f"Creating improvement plan (Attempt {final_state.get('enhancement_attempts', 1)}/3)...",
                        transcript_text,
                        enhanced_text,
                        linkedin_text,
                        verification_text,
                        improvement_plan_text,
                        research_results_text,
                        "",
                        state,
                        *loading_states
                    ]
                    
                    # Show researching state
                    loading_states, message = update_loading_state("research")
                    yield [
                        "",
                        f"Researching content (Attempt {final_state.get('enhancement_attempts', 1)}/3)...",
                        transcript_text,
                        enhanced_text,
                        linkedin_text,
                        verification_text,
                        improvement_plan_text,
                        research_results_text,
                        "",
                        state,
                        *loading_states
                    ]
                    
                    # Research content
                    state = research_content(state)
                    research_results_text = format_research_results(safe_json_loads(state.get("research_context", "{}")))
                    
                    # Show enhancing again state
                    loading_states, message = update_loading_state("enhance")
                    yield [
                        "",
                        f"Enhancing content again (Attempt {final_state.get('enhancement_attempts', 1)}/3)...",
                        transcript_text,
                        enhanced_text,
                        linkedin_text,
                        verification_text,
                        improvement_plan_text,
                        research_results_text,
                        "",
                        state,
                        *loading_states
                    ]
                    
                    # Enhance again
                    state = enhance_again(state)
                    enhanced_text = state["enhanced"]
                    
                    # Update LinkedIn post
                    state["enhanced"] = enhanced_text
                    linkedin_state = format_linkedin_post(state)
                    linkedin_text = linkedin_state["linkedin_post"]
                    
                    # Verify again
                    state["linkedin_post"] = linkedin_text
                    final_state = verify_content(state)
                    verification_text = format_verification_text(final_state.get("verification", {}))
                    improvement_plan_text = format_improvement_plan(final_state.get("improvement_plan", {}))
                    research_results_text = format_research_results(safe_json_loads(final_state.get("research_context", "{}")))
                
                # After research and enhancement, create improved LinkedIn post
                if final_state.get("needs_improvement", False):
                    # Show improved post loading state
                    loading_states, message = update_loading_state("improved")
                    yield [
                        "",
                        f"Creating improved LinkedIn post (Attempt {final_state.get('enhancement_attempts', 1)}/3)...",
                        transcript_text,
                        enhanced_text,
                        linkedin_text,
                        verification_text,
                        improvement_plan_text,
                        research_results_text,
                        message,  # improved linkedin (loading)
                        state,
                        *loading_states
                    ]
                    
                    # Create improved LinkedIn post
                    improved_state = format_linkedin_post(final_state)
                    improved_text = improved_state["linkedin_post"]
                    
                    # Update final state
                    final_state["improved_linkedin"] = improved_text
                
                # Complete
                loading_states, _ = update_loading_state("done")
                yield [
                    "",
                    "✅ Processing complete!",
                    transcript_text,
                    enhanced_text,
                    linkedin_text,
                    verification_text,
                    improvement_plan_text,
                    research_results_text,
                    final_state.get("improved_linkedin", "No improvements needed"),
                    final_state,
                    *loading_states
                ]
                
            except Exception as e:
                loading_states, _ = update_loading_state("done")
                yield [
                    f"⚠️ Error: {str(e)}",
                    "❌ Processing failed",
                    state.get("transcript", ""),
                    state.get("enhanced", ""),
                    state.get("linkedin_post", ""),
                    "",
                    "Error occurred during processing",
                    "Error occurred during processing",
                    "Error occurred during processing",
                    state,
                    *loading_states
                ]

        # Set up event handlers
        youtube_convert_btn.click(
            fn=process_with_loading,
            inputs=[video_url, current_state],
            outputs=[
                error,
                status,
                transcript,
                enhanced,
                linkedin,
                verification,
                improvement_plan,
                research_results,
                improved_linkedin,
                current_state,
                transcript_loading,
                enhanced_loading,
                linkedin_loading,
                verify_loading,
                plan_loading,
                research_loading,
                improved_loading
            ],
            show_progress=False
        )
        
        # Update error visibility
        error.change(
            lambda x: gr.update(visible=bool(x)),
            error,
            error
        )
    
    return demo

def print_graph():
    """Print ASCII representation of the workflow graph."""
    print("\nWorkflow Graph Visualization:")
    print("-----------------------------")
    print("""
    Main Workflow with Agentic Enhancement:
    [get_transcript] -> [enhance_content] -> [format_linkedin] -> [verify_content] -> [agent_decide] -> [END]
           |                      |                 |                  |                    |
           |                      |                 |                  |                    |
           v                      v                 v                  v                    v
        [ERROR] -> [END]      [ERROR] -> [END]   [ERROR] -> [END]   [ERROR] -> [END]    [ERROR] -> [END]
                                                                    |
                                                                    v
                                                              [needs_improvement]
                                                                    |
                                                                    v
                                                              [research_content] -> [enhance_again] -> [verify_content]
                                                                    |                      |                  |
                                                                    v                      v                  v
                                                              [ERROR] -> [END]        [ERROR] -> [END]    [ERROR] -> [END]
    """)
    print("-----------------------------\n")

def extract_text_from_webpage(url: str) -> str:
    """Extract main content text from a webpage."""
    try:
        # Use trafilatura for better content extraction
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_links=False, include_images=False)
            if text:
                return text.strip()
            
        # Fallback to basic BeautifulSoup extraction
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting webpage content: {str(e)}")

def process_youtube_video(video_url: str, model_name: str = "Shipmaster1/finetuned_mpnet_matryoshka_mnr"):
    """Process a YouTube video and store its content in the vector store using LangChain."""
    try:
        # Get video transcript
        transcript = get_transcript({"video_url": video_url})["transcript"]
        if not transcript:
            return None, "Failed to get video transcript"
            
        # Get video metadata
        video_info = {
            "video_id": extract_video_id(video_url),
            "title": "Untitled Video",
            "channel": "Unknown Channel",
            "url": video_url,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create document with metadata
        doc = Document(
            page_content=transcript,
            metadata=video_info
        )
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_documents([doc])
        
        # Store in Chroma using LangChain's abstraction
        collection = get_chroma_collection(model_name)
        collection.add_documents(chunks)
        
        return doc, "Successfully processed video"
        
    except Exception as e:
        return None, f"Error processing video: {str(e)}"

def process_webpage(url: str, model_name: str = "Shipmaster1/finetuned_mpnet_matryoshka_mnr"):
    """Process a webpage and store its content in the vector store using LangChain."""
    try:
        # Get webpage content
        content = extract_text_from_webpage(url)
        if not content:
            return None, "Failed to extract webpage content"
            
        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "url": url,
                "source": "webpage",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents([doc])
        
        # Store in Chroma using LangChain's abstraction
        collection = get_chroma_collection(model_name)
        collection.add_documents(chunks)
        
        return doc, "Successfully processed webpage"
        
    except Exception as e:
        return None, f"Error processing webpage: {str(e)}"

def agent_decide(state: ProcessState, progress=gr.Progress()) -> ProcessState:
    """Agent decides whether to enhance content further based on verification score and creates an improvement plan."""
    try:
        progress(0.95, desc="Analyzing content quality and planning improvements...")
        
        # Get verification score and attempts
        score = state.get("verification", {}).get("score", 0)
        attempts = state.get("enhancement_attempts", 0)
        feedback = state.get("verification", {}).get("feedback", "")
        
        # Initialize LLM for agentic decision making
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content strategist. Analyze the content quality and create an improvement plan.

Current Content:
{content}

Verification Results:
- Score: {score}
- Feedback: {feedback}
- Previous Attempts: {attempts}

Create a detailed improvement plan in JSON format:
{{
    "needs_improvement": boolean,
    "improvement_areas": [
        {{
            "area": string,
            "priority": number (1-5),
            "strategy": string,
            "research_focus": string
        }}
    ],
    "research_priorities": [
        {{
            "topic": string,
            "reason": string,
            "expected_impact": string
        }}
    ],
    "enhancement_strategy": {{
        "approach": string,
        "key_focus": string,
        "expected_improvements": [string]
    }}
}}

Consider:
1. Content quality and engagement
2. Information accuracy and completeness
3. Target audience needs
4. Previous enhancement attempts
5. Available research context"""),
            ("human", "Analyze this content and create an improvement plan.")
        ])
        
        chain = prompt | llm | StrOutputParser()
        plan = json.loads(chain.invoke({
            "content": state["enhanced"],
            "score": score,
            "feedback": feedback,
            "attempts": attempts
        }))
        
        # Update state with plan
        state["verification_score"] = score
        state["enhancement_attempts"] = attempts
        state["needs_improvement"] = plan["needs_improvement"]
        state["improvement_plan"] = plan
        
        # Create detailed status message
        if plan["needs_improvement"] and attempts < 3:
            status = f"🔄 Planning improvements (Attempt {attempts + 1}/3)\n"
            status += "Key focus areas:\n"
            for area in plan["improvement_areas"][:2]:  # Show top 2 priorities
                status += f"• {area['area']} (Priority: {area['priority']})\n"
            state["status"] = status
        else:
            if score >= 0.95:
                state["status"] = "✅ Content quality threshold met"
            else:
                state["status"] = "⚠️ Max enhancement attempts reached"
        
        return state
    except Exception as e:
        state["error"] = f"⚠️ Error in agent decision: {str(e)}"
        state["status"] = "❌ Failed to analyze content"
        return state

def research_content(state: ProcessState, progress=gr.Progress()) -> ProcessState:
    """Research additional context based on the improvement plan."""
    try:
        progress(0.96, desc="Researching based on improvement plan...")
        
        # Get improvement plan
        plan = state.get("improvement_plan", {})
        if not plan:
            raise Exception("No improvement plan found")
        
        # Initialize research results
        research_results = {
            "similar_content": [],
            "focused_research": {},
            "verification_feedback": state.get("verification", {}).get("feedback", "")
        }
        
        # Get similar content from vector store
        collection = get_chroma_collection()
        
        # Research each priority area
        for area in plan["improvement_areas"]:
            # Search for content related to this area
            similar_docs = collection.similarity_search(
                f"{area['area']} {area['research_focus']}",
                k=2
            )
            
            # Process with fine-tuned model
            area_features = model_manager.process_text("\n".join([doc.page_content for doc in similar_docs]))
            
            # Store research results
            research_results["focused_research"][area["area"]] = {
                "content": [doc.page_content for doc in similar_docs],
                "features": area_features,
                "priority": area["priority"],
                "strategy": area["strategy"]
            }
        
        # Research specific topics from research_priorities
        for topic in plan["research_priorities"]:
            topic_docs = collection.similarity_search(
                topic["topic"],
                k=1
            )
            if topic_docs:
                research_results["similar_content"].extend([doc.page_content for doc in topic_docs])
        
        # Store research results
        state["research_context"] = json.dumps(research_results)
        state["status"] = "✅ Research completed based on improvement plan"
        return state
    except Exception as e:
        state["error"] = f"⚠️ Error researching content: {str(e)}"
        state["status"] = "❌ Failed to research content"
        return state

def enhance_again(state: ProcessState, progress=gr.Progress()) -> ProcessState:
    """Enhance content using research and improvement plan."""
    try:
        progress(0.97, desc="Enhancing content based on research and plan...")
        
        # Get research context and improvement plan
        research_context = json.loads(state["research_context"])
        plan = state.get("improvement_plan", {})
        if not plan:
            raise Exception("No improvement plan found")
        
        # Initialize LLM for enhancement
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content enhancer. Improve the content based on the research and improvement plan while maintaining the original topic and key messages.

Current Content:
{content}

Improvement Plan:
{plan}

Research Results:
{research}

Enhancement Strategy:
{strategy}

Create enhanced content that:
1. Maintains the original topic and key messages
2. Addresses each improvement area according to its priority
3. Incorporates relevant research findings
4. Follows the enhancement strategy
5. Improves engagement and clarity
6. Keeps the same core subject matter and examples

Important:
- DO NOT change the main topic or subject matter
- DO NOT replace specific examples with generic ones
- DO NOT lose the original context or purpose
- DO NOT generate content about a different topic
- DO preserve and enhance the original message"""),
            ("human", "Enhance this content while maintaining its original topic and key messages.")
        ])
        
        chain = prompt | llm | StrOutputParser()
        enhanced = chain.invoke({
            "content": state["enhanced"],
            "plan": json.dumps(plan),
            "research": json.dumps(research_context),
            "strategy": json.dumps(plan["enhancement_strategy"])
        })
        
        # Update state
        state["enhanced"] = enhanced
        state["enhancement_attempts"] = state.get("enhancement_attempts", 0) + 1
        state["status"] = f"✅ Content enhanced with research (Attempt {state['enhancement_attempts']}/3)"
        return state
    except Exception as e:
        state["error"] = f"⚠️ Error enhancing content: {str(e)}"
        state["status"] = "❌ Failed to enhance content"
        return state

if __name__ == "__main__":
    print_graph()  # Print the graph visualization
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=True,
        show_api=False
    ) 