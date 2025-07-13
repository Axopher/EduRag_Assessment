# llm_config.py
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import HuggingFaceHub # Example for Hugging Face
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # NEW: For Gemini

# Load environment variables from .env file
load_dotenv()

# --- LLM and Embedding Model Configuration ---
# You can switch between OpenAI, Hugging Face, or Gemini models here.
# Ensure the respective API keys are set in your .env file.

# Choose your LLM provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower() # 'openai', 'huggingface', or 'gemini'

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-ada-002")

# Hugging Face Configuration (Example - adjust as needed)
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACE_LLM_REPO_ID = os.getenv("HUGGINGFACE_LLM_REPO_ID", "google/flan-t5-large")
HUGGINGFACE_EMBEDDING_MODEL_REPO_ID = os.getenv("HUGGINGFACE_EMBEDDING_MODEL_REPO_ID", "sentence-transformers/all-MiniLM-L6-v2")

# Gemini Configuration
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro") # or "gemini-1.5-pro-latest" etc.
GEMINI_EMBEDDING_MODEL_NAME = os.getenv("GEMINI_EMBEDDING_MODEL_NAME", "models/embedding-001")


def get_llm():
    """Initializes and returns the configured LLM."""
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return ChatOpenAI(model_name=OPENAI_MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.7)
    elif LLM_PROVIDER == "huggingface":
        if not HUGGINGFACEHUB_API_TOKEN:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set.")
        return HuggingFaceHub(repo_id=HUGGINGFACE_LLM_REPO_ID, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)
    elif LLM_PROVIDER == "gemini": # NEW: Gemini LLM
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.7)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

def get_embedding_model():
    """Initializes and returns the configured embedding model."""
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_NAME, api_key=OPENAI_API_KEY)
    elif LLM_PROVIDER == "huggingface":
        if not HUGGINGFACEHUB_API_TOKEN:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set.")
        # Note: LangChain's HuggingFaceEmbeddings might require specific setup
        # or a local model. For simplicity, we'll use OpenAI for embeddings in this example.
        # If you truly want Hugging Face embeddings, you might need to install 'sentence-transformers'
        # and use HuggingFaceInferenceEmbeddings or HuggingFaceEmbeddings(model_name=...)
        print("Warning: Using OpenAI embeddings even if LLM_PROVIDER is HuggingFace. Consider 'sentence-transformers' for HF embeddings.")
        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_NAME, api_key=OPENAI_API_KEY) # Fallback
    elif LLM_PROVIDER == "gemini": # NEW: Gemini Embedding
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        return GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

# --- Tutor Personas ---
PERSONAS = {
    "friendly": """You are a friendly and encouraging educational tutor. Your goal is to explain concepts clearly, patiently, and in an approachable manner. Use simple language and positive reinforcement. Always answer questions based on the provided context.""",
    "strict": """You are a strict and rigorous educational tutor. Your goal is to provide precise, factual, and concise answers. Avoid colloquialisms and focus directly on the academic content. Emphasize accuracy and depth. Always answer questions based on the provided context.""",
    "humorous": """You are a humorous and witty educational tutor. Your goal is to make learning fun by incorporating lighthearted jokes or playful analogies into your explanations. Keep the tone engaging and entertaining while still delivering accurate information. Always answer questions based on the provided context.""",
    "default": """You are an intelligent educational tutor. Your goal is to provide clear, concise, and accurate answers to student questions based on the provided context. If the context does not contain the answer, state that you cannot answer based on the provided information."""
}

def get_persona_prompt(persona_name: str) -> str:
    """Returns the system prompt for a given persona."""
    return PERSONAS.get(persona_name.lower(), PERSONAS["default"])
