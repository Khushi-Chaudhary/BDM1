import os
import time
import pickle
from sentence_transformers import SentenceTransformer
from app.file_utils import logger

# Global variables
EMBEDDINGS_CACHE = None
FILE_HASHES = {}  # Store file hashes and modification times to track changes
LAST_RELOAD_TIME = 0  # To track the last reload time
RELOAD_INTERVAL = 120  # Reload interval in seconds (e.g., 2 minutes)
CACHE_PATH = "embeddings_cache.pkl"  # Path for caching the embeddings

model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m")  # You can change the model name

def is_cache_valid():
    """Check if the cache file exists and is up to date."""
    if not os.path.exists(CACHE_PATH):
        return False
    cache_mod_time = os.path.getmtime(CACHE_PATH)
    current_time = time.time()
    return current_time - cache_mod_time < RELOAD_INTERVAL


def load_embeddings_from_cache():
    """Load the embeddings from the cache file."""
    with open(CACHE_PATH, 'rb') as cache_file:
        return pickle.load(cache_file)


def save_embeddings_to_cache(embeddings):
    """Save the embeddings to the cache file."""
    with open(CACHE_PATH, 'wb') as cache_file:
        pickle.dump(embeddings, cache_file)


def retrieve_all_documents(supabase):
    """Retrieve all documents (id and content) from the Supabase documents table."""
    response = supabase.table("documents").select("id", "content").execute()
    if response.status_code == 200:
        return response.data
    else:
        logger.error(f"Error retrieving documents from Supabase: {response.error}")
        return None


def create_embeddings(supabase):
    """
    Create embeddings for document texts from the 'documents' table and store them in the 'embeddings' table.
    """
    response = retrieve_all_documents(supabase)

    if not response:
        logger.error("No documents found in the 'documents' table.")
        return []

    new_embeddings = []
    for doc in response:
        document_id = doc["id"]
        text = doc["content"]
        embedding = model.encode([text])[0]  # Encoding a list of one document text, return the first result
        
        embeddings_to_store = {
            "document_id": document_id,  # Reference to the document
            "embedding": embedding.tolist()  # Store the embedding as a list (JSON-compatible)
        }

        # Optionally, store embeddings in Supabase (you can use Supabase's table to store these embeddings)
        response = supabase.table("embeddings").insert(embeddings_to_store).execute()

        #if response.status_code == 201 or response.status_code == 200:
        if response:
            new_embeddings.append(embedding)  # Append only the embedding (not the full data structure)
            logger.info(f"Embedding for document ID {document_id} stored successfully.")
        else:
            logger.error(f"Error storing embedding for document ID {document_id}: {response.error}")

    return new_embeddings

def reload_embeddings_if_needed(supabase):
    global EMBEDDINGS_CACHE, FILE_HASHES, LAST_RELOAD_TIME
    current_time = time.time()

    # If the embeddings cache is still valid, return it
    if current_time - LAST_RELOAD_TIME < RELOAD_INTERVAL and EMBEDDINGS_CACHE is not None:
        logger.info("Using cached embeddings.")
        return EMBEDDINGS_CACHE

    # If cache is valid, load and return it
    if is_cache_valid():
        EMBEDDINGS_CACHE = load_embeddings_from_cache()
        logger.info("Loaded embeddings from cache.")
        return EMBEDDINGS_CACHE

    # Retrieve embeddings from Supabase if cache is not valid
    embeddings = []
    supabase_documents = retrieve_all_documents(supabase)
    if supabase_documents is None:
        logger.error("Error retrieving documents from Supabase.")
        return EMBEDDINGS_CACHE

    # Generate embeddings for the retrieved documents
    embeddings = create_embeddings(supabase)

    if not embeddings:
        logger.warning("No new embeddings generated.")
        return EMBEDDINGS_CACHE

    # Save the embeddings to cache
    save_embeddings_to_cache(embeddings)
    LAST_RELOAD_TIME = current_time

    logger.info("Embeddings reloaded and saved to cache.")

    return embeddings
