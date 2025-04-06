from pinecone import Pinecone, Index
import os

# Replace with your API key and environment
api_key = os.environ.get("PINECONE_API_KEY")
environment = os.environ.get("PINECONE_ENVIRONMENT")

pinecone = Pinecone(api_key=api_key, environment=environment)

index_name = "your-index-name" # replace with your index name

# if the index does not exist
if index_name not in pinecone.list_indexes():
    # we create the index
    pinecone.create_index(
        name=index_name,
        dimension=384,  # Replace with the embedding dimension (all-MiniLM-L6-v2 is 384)
        metric="cosine", # You can use "cosine" for cosine similarity
    )
    
# Connect to the index
index = pinecone.Index(index_name)