import os
import hashlib
from app.file_utils import load_hidden_documents, logger
from app.vector_store import create_emebeddings
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase URL and key are not set in the environment variables.")
    raise ValueError("Supabase URL and key are not set in the environment variables.")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def store_documents(supabase, documents):
    """Store document hashes in Supabase."""
    try:
        documents_to_insert = []
        for doc_text in documents:
            content_hash = hashlib.sha256(doc_text.encode()).hexdigest()
            existing = supabase.table("documents").select("id").eq("hash", content_hash).execute()
            #print(f"Type of response from supabase is {type(existing)}")
            if existing is None:
                logger.error(f"Error checking existing document: {existing.error}")
                continue
            if existing.data:
                logger.info(f"Document with hash '{content_hash[:8]}...' already exists, skipping.")
                continue
            documents_to_insert.append({"hash": content_hash})

        if documents_to_insert:  # Only insert if there are new documents
            response = supabase.table("documents").insert(documents_to_insert).execute()
            if response is None:
                logger.error(f"Error storing documents: {response.error}")
            else:
                logger.info(f"{len(documents_to_insert)} new documents (hashes) stored successfully.")

    except Exception as e:
        logger.exception(f"Error interacting with Supabase: {e}")

documents=load_hidden_documents(directory="hidden_docs")
store_documents(supabase, documents)


def retrieve_all_documents(supabase):
    """Retrieves all document hashes and texts from Supabase."""
    try:
        response = supabase.table("documents").select("hash", "text").execute()  # get text as well
        if response is None:
            logger.error(f"Error retrieving all documents: {response.error}")
            return None
        return {doc["hash"]: doc["text"] for doc in response.data}  # create dict for efficient lookup
    except Exception as e:
        logger.exception(f"Error interacting with Supabase: {e}")
        return None

def store_embeddings(supabase, embedding_data):
    """
    Store embeddings in Supabase with a check for duplicates.
    
    :param supabase: Supabase client instance.
    :param embedding_data: List of dictionaries with keys 'hash', 'text', and 'embedding'.
    :return: True if storage was successful, False otherwise.
    """
    try:
        new_embeddings = []
        
        for item in embedding_data:
            # Check if the embedding with the same hash already exists
            existing = supabase.table("embeddings").select("id").eq("hash", item["hash"]).execute()
            if existing.data:
                logger.info(f"Embedding for hash '{item['hash'][:8]}...' already exists, skipping.")
                continue
            
            new_embeddings.append(item)
        
        if new_embeddings:  # Only insert if there are new embeddings to store
            response = supabase.table("embeddings").insert(new_embeddings).execute()
            if response.error is not None:
                logger.error(f"Error storing new embeddings: {response.error}")
                return False
            logger.info(f"{len(new_embeddings)} new embeddings stored successfully.")
        else:
            logger.info("No new embeddings to store.")
        
        return True

    except Exception as e:
        logger.exception(f"Error interacting with Supabase while storing embeddings: {e}")
        return False


emebddings=create_emebeddings(documents)
store_embeddings(supabase, emebddings)

def retrieve_relevant_texts(supabase, query_embedding, top_k=5):
    """Retrieve the most relevant texts based on a query embedding."""
    try:
        response = supabase.rpc("match_embeddings", {
            "query_embedding": query_embedding,
            "top_k": top_k
        }).execute()
        if response is None:
            print(f"Error retrieving texts: {response.error}")
        else:
            return response.data
    except Exception as e:
        print(f"Error interacting with Supabase: {e}")
        return []

def store_chat_history(supabase, chat_id, chat_history):
    """Store chat history in Supabase."""
    try:
        response = supabase.table("chat_history").insert({
            "chat_id": chat_id,
            "history": chat_history
        }).execute()
        if response is None:
            print(f"Error storing chat history: {response.error}")
        else:
            print("Chat history stored successfully.")
    except Exception as e:
        print(f"Error interacting with Supabase: {e}")

def retrieve_chat_history(supabase, chat_id):
    """Retrieve chat history from Supabase."""
    try:
        response = supabase.table("chat_history").select("history").eq("chat_id", chat_id).execute()
        if response.error is not None:
            print(f"Error retrieving chat history: {response.error}")
            return []
        elif response.data:
            return response.data[0]["history"]
        else:
            print("No chat history found for the given chat_id.")
            return []
    except Exception as e:
        print(f"Error interacting with Supabase: {e}")
        return []
