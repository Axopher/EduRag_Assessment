# rag_pipeline.py
import os
import faiss
import numpy as np
from typing import List, Tuple
from sqlalchemy.orm import Session
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import models, llm_config, database

# Global variables for FAISS index and content mapping
# In a real-world scenario, these should be loaded from persistent storage
# and updated incrementally. For this example, we'll build it on startup.
vector_store = None
content_id_map = {} # Maps FAISS index to content ID

def initialize_vector_store(db: Session):
    """
    Initializes the FAISS vector store with existing content from the database.
    This should be called once on application startup.
    """
    global vector_store, content_id_map

    print("Initializing vector store...")
    embedding_model = llm_config.get_embedding_model()
    all_content = db.query(models.Content).all()

    # Determine embedding dimension dynamically or set a default for common models
    embedding_dimension = 768 # Default for Gemini's 'models/embedding-001' or common sentence transformers
    if llm_config.LLM_PROVIDER == "openai":
        embedding_dimension = 1536 # Default for OpenAI's 'text-embedding-ada-002'

    try:
        # Attempt to get a dummy embedding to accurately determine dimension
        dummy_embedding = embedding_model.embed_query("test")
        embedding_dimension = len(dummy_embedding)
        print(f"Dynamically determined embedding dimension: {embedding_dimension}")
    except Exception as e:
        print(f"Could not get embedding dimension from model, using default {embedding_dimension}: {e}")
        # Fallback to a reasonable default based on provider, if dynamic check fails

    if not all_content:
        print("No content found in database to initialize vector store. Creating an empty FAISS index.")
        vector_store = faiss.IndexFlatL2(embedding_dimension)
        return

    texts = [c.content_text for c in all_content]
    content_ids = [c.id for c in all_content]

    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} content pieces...")
    embeddings = embedding_model.embed_documents(texts)
    embeddings_np = np.array(embeddings).astype('float32')

    # Ensure the determined embedding dimension matches the generated embeddings
    if embeddings_np.shape[1] != embedding_dimension:
        print(f"Warning: Initial embedding dimension ({embedding_dimension}) mismatch with generated embeddings ({embeddings_np.shape[1]}). Adjusting.")
        embedding_dimension = embeddings_np.shape[1]


    # Create FAISS index
    vector_store = faiss.IndexFlatL2(embedding_dimension)
    vector_store.add(embeddings_np)

    # Create content ID map
    content_id_map = {i: content_ids[i] for i in range(len(content_ids))}
    print(f"Vector store initialized with {len(texts)} documents.")

def add_content_to_vector_store(db: Session, content_entry: models.Content):
    """
    Adds a new content entry to the existing FAISS vector store.
    """
    global vector_store, content_id_map

    embedding_model = llm_config.get_embedding_model()
    text = content_entry.content_text
    content_id = content_entry.id

    if vector_store is None:
        # If vector store is not initialized (e.g., first content upload), initialize it
        print("Vector store not initialized, initializing now with new content.")
        initialize_vector_store(db)
        # After initialization, the content_entry might already be added if it was the first.
        # We need to check if it's already there to avoid duplicates.
        if content_id not in content_id_map.values():
             # If not added by initialize_vector_store (e.g., if it was empty before), add it now.
             embedding = embedding_model.embed_query(text)
             embedding_np = np.array([embedding]).astype('float32')
             current_index_size = vector_store.ntotal
             vector_store.add(embedding_np)
             content_id_map[current_index_size] = content_id
             print(f"Added content ID {content_id} to vector store at index {current_index_size} after initial empty state.")
        return

    # Generate embedding for the new content
    embedding = embedding_model.embed_query(text)
    embedding_np = np.array([embedding]).astype('float32')

    # Add to FAISS index
    current_index_size = vector_store.ntotal
    vector_store.add(embedding_np)

    # Update content ID map
    content_id_map[current_index_size] = content_id
    print(f"Added content ID {content_id} to vector store at index {current_index_size}.")

def retrieve_relevant_content(query: str, k: int = 3) -> List[Tuple[str, int]]:
    """
    Retrieves the top-k most semantically similar content pieces from the vector store.
    Returns a list of (content_text, content_id) tuples.
    """
    global vector_store, content_id_map

    if vector_store is None or vector_store.ntotal == 0:
        print("Vector store is empty or not initialized. Cannot retrieve content.")
        return []

    embedding_model = llm_config.get_embedding_model()
    query_embedding = embedding_model.embed_query(query)
    query_embedding_np = np.array([query_embedding]).astype('float32')

    # Perform similarity search
    distances, indices = vector_store.search(query_embedding_np, k)

    retrieved_data = []
    db_session = next(database.get_db()) # Get a new DB session for retrieval

    for i, idx in enumerate(indices[0]):
        if idx in content_id_map:
            content_id = content_id_map[idx]
            content_entry = db_session.query(models.Content).filter(models.Content.id == content_id).first()
            if content_entry:
                retrieved_data.append((content_entry.content_text, content_entry.id))
        else:
            print(f"Warning: FAISS index {idx} not found in content_id_map.")
    db_session.close()
    return retrieved_data

def generate_rag_answer(query: str, persona_name: str) -> Tuple[str, List[int]]:
    """
    Generates an answer using the RAG pipeline.
    Returns the AI answer and a list of IDs of the retrieved content.
    """
    llm = llm_config.get_llm()
    system_prompt = llm_config.get_persona_prompt(persona_name)

    # Retrieve relevant content
    retrieved_content_tuples = retrieve_relevant_content(query)
    retrieved_texts = [text for text, _ in retrieved_content_tuples]
    retrieved_ids = [id for _, id in retrieved_content_tuples]

    if not retrieved_texts:
        return "I couldn't find relevant information in my knowledge base to answer your question. Please try rephrasing or ask about a different topic.", []

    # Format retrieved documents for the prompt
    context = "\n\n".join(retrieved_texts)

    # LangChain RAG chain
    template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\nContext: {context}"),
            ("human", "{question}"),
        ]
    )

    # Simple RAG chain without a dedicated retriever from LangChain
    # We are doing retrieval manually and passing the context.
    chain = (
        {"context": lambda x: context, "question": RunnablePassthrough()}
        | template
        | llm
        | StrOutputParser()
    )

    print(f"Generating RAG answer for query: {query} with persona: {persona_name}")
    answer = chain.invoke(query)
    return answer, retrieved_ids
