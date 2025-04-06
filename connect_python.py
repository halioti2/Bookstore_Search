import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, Index

load_dotenv()

# Replace with your API key and environment
api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key)

# Create a dense index with integrated embedding
index_name = "dense-index"

# if the index does not exist
#if index_name not in pinecone.list_indexes(): -old
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
#     { "_id": "rec21", "chunk_text": "The Statue of Liberty was a gift from France to the United States.", "category": "history" },
#     { "_id": "rec22", "chunk_text": "Coffee contains caffeine, a natural stimulant.", "category": "food science" },
#     { "_id": "rec23", "chunk_text": "Thomas Edison invented the practical electric light bulb.", "category": "inventions" },
#     { "_id": "rec24", "chunk_text": "The moon influences ocean tides due to gravitational pull.", "category": "astronomy" },
#     { "_id": "rec25", "chunk_text": "DNA carries genetic information for all living organisms.", "category": "biology" },
#     { "_id": "rec26", "chunk_text": "Rome was once the center of a vast empire.", "category": "history" },
#     { "_id": "rec27", "chunk_text": "The Wright brothers pioneered human flight in 1903.", "category": "inventions" },
#     { "_id": "rec28", "chunk_text": "Bananas are a good source of potassium.", "category": "nutrition" },
#     { "_id": "rec29", "chunk_text": "The stock market fluctuates based on supply and demand.", "category": "economics" },
#     { "_id": "rec30", "chunk_text": "A compass needle points toward the magnetic north pole.", "category": "navigation" },
#     { "_id": "rec31", "chunk_text": "The universe is expanding, according to the Big Bang theory.", "category": "astronomy" },
#     { "_id": "rec32", "chunk_text": "Elephants have excellent memory and strong social bonds.", "category": "biology" },
#     { "_id": "rec33", "chunk_text": "The violin is a string instrument commonly used in orchestras.", "category": "music" },
#     { "_id": "rec34", "chunk_text": "The heart pumps blood throughout the human body.", "category": "biology" },
#     { "_id": "rec35", "chunk_text": "Ice cream melts when exposed to heat.", "category": "food science" },
#     { "_id": "rec36", "chunk_text": "Solar panels convert sunlight into electricity.", "category": "technology" },
#     { "_id": "rec37", "chunk_text": "The French Revolution began in 1789.", "category": "history" },
#     { "_id": "rec38", "chunk_text": "The Taj Mahal is a mausoleum built by Emperor Shah Jahan.", "category": "history" },
#     { "_id": "rec39", "chunk_text": "Rainbows are caused by light refracting through water droplets.", "category": "physics" },
#     { "_id": "rec40", "chunk_text": "Mount Everest is the tallest mountain in the world.", "category": "geography" },
#     { "_id": "rec41", "chunk_text": "Octopuses are highly intelligent marine creatures.", "category": "biology" },
#     { "_id": "rec42", "chunk_text": "The speed of sound is around 343 meters per second in air.", "category": "physics" },
#     { "_id": "rec43", "chunk_text": "Gravity keeps planets in orbit around the sun.", "category": "astronomy" },
#     { "_id": "rec44", "chunk_text": "The Mediterranean diet is considered one of the healthiest in the world.", "category": "nutrition" },
#     { "_id": "rec45", "chunk_text": "A haiku is a traditional Japanese poem with a 5-7-5 syllable structure.", "category": "literature" },
#     { "_id": "rec46", "chunk_text": "The human body is made up of about 60% water.", "category": "biology" },
#     { "_id": "rec47", "chunk_text": "The Industrial Revolution transformed manufacturing and transportation.", "category": "history" },
#     { "_id": "rec48", "chunk_text": "Vincent van Gogh painted Starry Night.", "category": "art" },
#     { "_id": "rec49", "chunk_text": "Airplanes fly due to the principles of lift and aerodynamics.", "category": "physics" },
#     { "_id": "rec50", "chunk_text": "Renewable energy sources include wind, solar, and hydroelectric power.", "category": "energy" }
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

# # Print the results
# for hit in results['result']['hits']:
#     print(f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}")

# Search the dense index and rerank results
reranked_results = dense_index.search(
    namespace="example-namespace",
    query={
        "top_k": 10,
        "inputs": {
            'text': query
        }
    },
    rerank={
        "model": "bge-reranker-v2-m3",
        "top_n": 10,
        "rank_fields": ["chunk_text"]
    }   
)

# Print the reranked results
for hit in reranked_results['result']['hits']:
    print(f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}")

# Delete the index
pc.delete_index(index_name)