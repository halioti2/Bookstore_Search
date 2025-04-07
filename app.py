import os
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from pinecone import Pinecone, Index
import json
import uuid

load_dotenv()

# Replace with your API key and environment
api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key)

app = Flask(__name__)

@app.route('/') # This means, when you visit the homepage (/)
def hello_world():
    return "Hello, World!"

@app.route('/reset_book', methods=['POST'])
def reset_book():
    try:
        # Delete the index
        pc.delete_index(index_name)
        
        # Recreate the index if it does not exist
        if not pc.has_index(index_name):
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "chunk_text"}
                }
            )
            print(f"Index '{index_name}' created.")
        else:
            print(f"Index '{index_name}' already exists.")

        # Wait for the index to stabilize
        time.sleep(10)

        return jsonify({"message": "Index reset successfully."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_book', methods=['POST'])
def add_book():
    try:
        # Parse the incoming JSON request
        book_data = request.get_json()
        
        # Validate required fields
        required_fields = ["_id", "title", "author", "isbn", "description", "category"]
        for field in required_fields:
            if field not in book_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Prepare the record for Pinecone
        records = [ {
            "_id": book_data["_id"],
            "chunk_text": f"Title: {book_data['title']}. Author: {book_data['author']}. ISBN: {book_data['isbn']}. Description: {book_data['description']}",
            "category": book_data["category"]
        } ]

        # Target the index
        dense_index = pc.Index(index_name)

        # Upsert the records into a namespace
        dense_index.upsert_records("example-namespace", records)

        # Wait for the upserted vectors to be indexed
        import time
        time.sleep(10)

        # View stats for the index
        stats = dense_index.describe_index_stats()
        print(f"test")
        print(stats)

        return jsonify({"message": "Book added successfully."}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Create a dense index with integrated embedding
index_name = "dense-index"

# if the index does not exist
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")

# Connect to the index
index = pc.Index(index_name)
print(f"Successfully connected to Pinecone index: {index_name}")

if __name__ == '__main__':
    app.run(debug=True)