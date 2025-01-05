import os
import time
import pickle
import hashlib
from app.data import supabase, retrieve_all_documents, store_embeddings
from app.file_utils import logger
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

# Global variables
VECTOR_STORE = None
FILE_HASHES = {}  # Store file hashes and modification times to track changes
LAST_RELOAD_TIME = 0  # To track the last reload time
RELOAD_INTERVAL = 120  # Reload interval in seconds (e.g., 2 minutes)
CACHE_PATH = "vector_store_cache.pkl"  # Path for caching the vector store

model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m")


def generate_text_hash(text):
    """
    Generate a uniform SHA-256 hash for a given text.
    Ensures consistent encoding and hashing across the application.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_file_hash(file_path):
    """Generate SHA-256 hash of the file content."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def get_file_mod_time(file_path):
    """Get the modification time of a specific file."""
    return os.path.getmtime(file_path)


def create_embeddings(document_texts):
    """
    Create embeddings for document texts, checking for existing embeddings in Supabase.
    Only new embeddings are generated and stored.
    """
    new_embeddings = []
    embeddings_to_store = []

    for text in document_texts:
        # Generate a unique hash for the document text to check for duplicates
        content_hash = generate_text_hash(text)

        # Check if the embedding already exists in Supabase
        existing = supabase.table("embeddings").select("id").eq("hash", content_hash).execute()

        if existing.data:
            logger.info(f"Embedding for text '{text[:30]}...' already exists, skipping.")
        else:
            # If no existing embedding is found, generate and store it
            logger.info(f"Generating new embedding for text '{text[:30]}...'")
            embedding = model.encode([text])[0]  # Encode text and get the first (only) result
            embeddings_to_store.append({"hash": content_hash, "text": text, "embedding": embedding.tolist()})
            new_embeddings.append(embedding)

    # Store new embeddings in Supabase if there are any
    if embeddings_to_store:
        response = supabase.table("embeddings").insert(embeddings_to_store).execute()
        if response.error is not None:
            logger.error(f"Error storing new embeddings: {response.error}")
        else:
            logger.info(f"{len(embeddings_to_store)} new embeddings stored successfully.")

    return new_embeddings


def create_vector_store(embeddings):
    """Create vector store from pre-existing embeddings."""
    return FAISS.from_embeddings(embeddings, model)


def is_cache_valid():
    """Check if the cache file exists and is up to date."""
    if not os.path.exists(CACHE_PATH):
        return False
    cache_mod_time = os.path.getmtime(CACHE_PATH)
    current_time = time.time()
    if current_time - cache_mod_time < RELOAD_INTERVAL:
        return True
    return False


def load_vector_store_from_cache():
    """Load the vector store from the cache file."""
    with open(CACHE_PATH, 'rb') as cache_file:
        return pickle.load(cache_file)


def save_vector_store_to_cache(vector_store):
    """Save the vector store to the cache file."""
    with open(CACHE_PATH, 'wb') as cache_file:
        pickle.dump(vector_store, cache_file)


def reload_vector_store_if_needed(supabase, directory="hidden_docs"):
    global VECTOR_STORE, FILE_HASHES, LAST_RELOAD_TIME
    current_time = time.time()

    if current_time - LAST_RELOAD_TIME < RELOAD_INTERVAL and VECTOR_STORE is not None:
        return VECTOR_STORE

    if is_cache_valid():
        VECTOR_STORE = load_vector_store_from_cache()
        print("Loaded vector store from cache.")
        return VECTOR_STORE

    current_file_data = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            content_hash = get_file_hash(file_path)
            if content_hash:
                current_file_data[content_hash] = filename
            else:
                logger.warning(f"Could not hash file: {filename}")
                return VECTOR_STORE

    supabase_documents = retrieve_all_documents(supabase)
    if supabase_documents is None:
        print("Error retrieving documents from Supabase.")
        return VECTOR_STORE

    hashes_to_reload = [
        file_hash
        for file_hash in current_file_data
        if file_hash not in FILE_HASHES or file_hash not in supabase_documents
    ]
    if hashes_to_reload:
        print(f"Reloading files based on content changes: {hashes_to_reload}")
        try:
            document_texts = []
            for file_hash, filename in current_file_data.items():
                if file_hash in hashes_to_reload:
                    file_path = os.path.join(directory, filename)
                    # Try multiple encodings in case of UTF-8 errors
                    document_text = None
                    encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
                    for encoding in encodings:
                        try:
                            with open(file_path, "r", encoding=encoding) as file:
                                document_text = file.read()
                            break  # If successful, exit loop
                        except UnicodeDecodeError:
                            logger.warning(f"Failed to decode {filename} using {encoding} encoding.")
                        except Exception as e:
                            logger.error(f"Error reading {filename}: {e}")
                            break

                    if document_text is not None:
                        document_texts.append(document_text)
                    else:
                        logger.warning(f"Could not read file {filename} after trying all encodings.")
                        continue  # Skip to next file if decoding fails

            if not document_texts:
                logger.warning("No documents found to create vector store")
                return VECTOR_STORE

            embeddings = create_embeddings(document_texts)
            if not embeddings:
                logger.warning("No new embeddings generated.")
                return VECTOR_STORE

            VECTOR_STORE = create_vector_store(embeddings)
            FILE_HASHES = {generate_text_hash(text): text for text in document_texts}
            save_vector_store_to_cache(VECTOR_STORE)
            LAST_RELOAD_TIME = current_time
        except Exception as e:
            print(f"Error while reloading vector store: {e}")
            return VECTOR_STORE
    else:
        print("No content changes detected. Using cached vector store.")

    return VECTOR_STORE


