import os
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from pinecone import Pinecone, Index
import json
import uuid
import requests # Add requests library for API calls

load_dotenv()

# Replace with your API key and environment
api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key)

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/reset_book', methods=['POST'])
def reset_book():
    try:
                # Target the index
        dense_index = pc.Index(index_name)
        
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

        # View stats for the index
        stats = dense_index.describe_index_stats()
        print(f"test")
        print(stats)
        return jsonify({"message": "Books reset successfully."}), 201
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

@app.route('/search', methods=['GET'])
def search():
    # 1. Get the search query from the URL
    query = request.args.get('q')
    
    if not query:
        return jsonify({"error": "Missing query parameter 'q'."}), 400

    # Target the index
    dense_index = pc.Index(index_name)

    try:
        # Perform the search in Pinecone
        results = dense_index.search(
            namespace="example-namespace",
            query={
                "top_k": 5,  # Fetch top 5 results
                "inputs": {
                    "text": query
                }
            }
        )

        # Check if there are any hits
        hits = results.get('result', {}).get('hits', [])
        if not hits:
            return jsonify({"results": [], "message": "No results found"}), 404

        # Process all hits
        formatted_results = []
        for hit in hits:
            formatted_results.append({
                "id": hit['_id'],
                "score": round(hit['_score'], 2),
                "text": hit['fields']['chunk_text'],
                "category": hit['fields']['category']
            })

        # Return the list of results as JSON
        return jsonify({"results": formatted_results}), 200

    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/isbn_lookup', methods=['GET'])
def isbn_lookup():
    isbn = request.args.get('isbn')
    if not isbn:
        return jsonify({"error": "Missing query parameter 'isbn'."}), 400

    # Basic ISBN format check (simple length check, not full validation)
    if not (len(isbn) == 10 or len(isbn) == 13) or not isbn.isdigit():
         # Allow ISBNs with hyphens by removing them first
        isbn_cleaned = isbn.replace('-', '')
        if not (len(isbn_cleaned) == 10 or len(isbn_cleaned) == 13) or not isbn_cleaned.isdigit():
            return jsonify({"error": "Invalid ISBN format. Must be 10 or 13 digits (hyphens optional)."}), 400
        isbn = isbn_cleaned # Use the cleaned version

    api_url = f"https://openlibrary.org/isbn/{isbn}.json"

    try:
        response = requests.get(api_url)
        
        # Check if the book was found (status code 200)
        if response.status_code == 200:
            return jsonify(response.json()), 200
        # Handle cases where the ISBN is not found (status code 404)
        elif response.status_code == 404:
            return jsonify({"error": f"No book found for ISBN: {isbn}"}), 404
        # Handle other potential API errors
        else:
            return jsonify({"error": f"Open Library API error. Status code: {response.status_code}", "details": response.text}), response.status_code

    except requests.exceptions.RequestException as e:
        # Handle network errors (e.g., connection timeout)
        return jsonify({"error": "Could not connect to Open Library API.", "details": str(e)}), 503 # Service Unavailable
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Error during ISBN lookup: {e}")
        return jsonify({"error": "An unexpected error occurred during ISBN lookup.", "details": str(e)}), 500


        # Print the results
        # for hit in results['result']['hits']:
        #     print(f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}")

    

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

# record setup
records = [
    { "_id": "book_001", "chunk_text": "Title: The Lord of the Rings. Author: J.R.R. Tolkien. ISBN: 978-0618260214. Description: A classic fantasy tale of good versus evil.", "category": "fantasy" },
    { "_id": "book_002", "chunk_text": "Title: Pride and Prejudice. Author: Jane Austen. ISBN: 978-0141439518. Description: A romance novel set in the English countryside.", "category": "classic" },
    { "_id": "book_003", "chunk_text": "Title: 1984. Author: George Orwell. ISBN: 978-0451524935. Description: A dystopian novel about totalitarianism.", "category": "dystopian" },
    { "_id": "book_004", "chunk_text": "Title: To Kill a Mockingbird. Author: Harper Lee. ISBN: 978-0446310757. Description: A novel about racial injustice in the American South.", "category": "classic" },
    { "_id": "book_005", "chunk_text": "Title: The Great Gatsby. Author: F. Scott Fitzgerald. ISBN: 978-0743273565. Description: A novel about the Roaring Twenties.", "category": "classic" },
    { "_id": "book_006", "chunk_text": "Title: The Hitchhiker's Guide to the Galaxy. Author: Douglas Adams. ISBN: 978-0345391803. Description: A comedic science fiction adventure.", "category": "science fiction" },
    { "_id": "book_007", "chunk_text": "Title: Dune. Author: Frank Herbert. ISBN: 978-0441172719. Description: A science fiction epic on a desert planet.", "category": "science fiction" },
    { "_id": "book_008", "chunk_text": "Title: The Martian. Author: Andy Weir. ISBN: 978-0553417765. Description: A science fiction survival story.", "category": "science fiction" },
    { "_id": "book_009", "chunk_text": "Title: And Then There Were None. Author: Agatha Christie. ISBN: 978-0062073483. Description: A classic murder mystery.", "category": "mystery" },
    { "_id": "book_010", "chunk_text": "Title: The Girl with the Dragon Tattoo. Author: Stieg Larsson. ISBN: 978-0307269751. Description: A crime thriller.", "category": "thriller" },
    { "_id": "book_011", "chunk_text": "Title: Gone with the Wind. Author: Margaret Mitchell. ISBN: 978-1451635621. Description: A historical romance set during the Civil War.", "category": "historical romance" },
    { "_id": "book_012", "chunk_text": "Title: The Secret Garden. Author: Frances Hodgson Burnett. ISBN: 978-0140300582. Description: A children's classic about a hidden garden.", "category": "childrens" },
    { "_id": "book_013", "chunk_text": "Title: Moby Dick. Author: Herman Melville. ISBN: 978-0553213119. Description: A novel about a whale hunt.", "category": "classic" },
    { "_id": "book_014", "chunk_text": "Title: War and Peace. Author: Leo Tolstoy. ISBN: 978-0679783268. Description: A historical novel set during the Napoleonic Wars.", "category": "historical" },
    { "_id": "book_015", "chunk_text": "Title: The Catcher in the Rye. Author: J.D. Salinger. ISBN: 978-0316769488. Description: A coming-of-age story.", "category": "coming-of-age" },
    { "_id": "book_016", "chunk_text": "Title: The Picture of Dorian Gray. Author: Oscar Wilde. ISBN: 978-0486277456. Description: A philosophical novel.", "category": "classic" },
    { "_id": "book_017", "chunk_text": "Title: Dracula. Author: Bram Stoker. ISBN: 978-0553213133. Description: A gothic horror novel.", "category": "horror" },
    { "_id": "book_018", "chunk_text": "Title: Frankenstein. Author: Mary Shelley. ISBN: 978-0486282115. Description: A science fiction horror novel.", "category": "horror" },
    { "_id": "book_019", "chunk_text": "Title: The Handmaid's Tale. Author: Margaret Atwood. ISBN: 978-0385490818. Description: A dystopian novel about a totalitarian society.", "category": "dystopian" },
    { "_id": "book_020", "chunk_text": "Title: The Hobbit. Author: J.R.R. Tolkien. ISBN: 978-0618260207. Description: A children's fantasy novel.", "category": "fantasy" }
]

# Target the index
dense_index = pc.Index(index_name)

# Upsert the records into a namespace
dense_index.upsert_records("example-namespace", records)

# Wait for the upserted vectors to be indexed
import time
time.sleep(10)

# View stats for the index
stats = dense_index.describe_index_stats()
print(stats)

# Define the query
query = "Lord of the Rings"

# Search the dense index
results = dense_index.search(
    namespace="example-namespace",
    query={
        "top_k": 10,
        "inputs": {
            'text': query
        }
    }
)

# Print the results
for hit in results['result']['hits']:
    print(f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}")

# # Delete the index
# pc.delete_index(index_name)

if __name__ == '__main__':
    app.run(debug=True)
