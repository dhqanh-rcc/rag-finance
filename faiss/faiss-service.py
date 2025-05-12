# from flask import Flask, request, jsonify
# import pandas as pd
# import os
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS

# app = Flask(__name__)

# # Load environment variables
# embeddings = OpenAIEmbeddings(openai_api_key="")
# filePath = "/Users/dhquanganh/Documents/dhquanganh/CPAV/rag-skupper/faiss/all_stocks_5yr.csv"
# # Global vector store
# vector_store = None

# def load_and_process_csv(file_path):
#     """Load and process CSV into text chunks"""
#     data = pd.read_csv(file_path).head(5)
#     data['text'] = data.apply(
#         lambda row: f"Stock {row['Name']} on date {row['date']} opening price {row['open']} closing price {row['close']}.",
#         axis=1
#     )
#     return data['text'].tolist()

# # Initialize data outside of any route (at startup)
# def initialize_vector_store():
#     """Load and embed data at startup"""
#     global vector_store
#     file_path = filePath
#     texts = load_and_process_csv(file_path)
#     vector_store = FAISS.from_texts(texts, embeddings)
#     print("✅ FAISS Vector Store loaded and ready.")

# # Initialize the vector store immediately
# initialize_vector_store()

# @app.route("/query", methods=["POST"])
# def query():
#     """API endpoint to retrieve similar documents from FAISS"""
#     data = request.json
#     question = data.get("question")

#     if not question:
#         return jsonify({"error": "No question provided"}), 400
        
#     if vector_store is None:
#         return jsonify({"error": "Vector store not initialized"}), 500

#     results = vector_store.similarity_search(question, k=5)
#     return jsonify({"results": [doc.page_content for doc in results]})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


from flask import Flask, request, jsonify
import pandas as pd
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

app = Flask(__name__)

# File path
filePath = "all_stocks_5yr.csv"
# Global vector store
vector_store = None

def load_and_process_csv(file_path):
    """Load and process CSV into text chunks"""
    data = pd.read_csv(file_path).head(5)
    data['text'] = data.apply(
        lambda row: f"Stock {row['Name']} on date {row['date']} opening price {row['open']} closing price {row['close']}.",
        axis=1
    )
    return data['text'].tolist()

# Initialize data outside of any route (at startup)
def initialize_vector_store():
    """Load and embed data at startup"""
    global vector_store
    try:
        # Using a free HuggingFace model for embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        file_path = filePath
        texts = load_and_process_csv(file_path)
        vector_store = FAISS.from_texts(texts, embeddings)
        print("✅ FAISS Vector Store loaded and ready.")
    except Exception as e:
        print(f"❌ Failed to initialize vector store: {str(e)}")

# Initialize the vector store immediately
initialize_vector_store()

@app.route("/query", methods=["POST"])
def query():
    """API endpoint to retrieve similar documents from FAISS"""
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400
        
    if vector_store is None:
        return jsonify({"error": "Vector store not initialized"}), 500

    try:
        results = vector_store.similarity_search(question, k=5)
        return jsonify({"results": [doc.page_content for doc in results]})
    except Exception as e:
        return jsonify({"error": f"Error processing query: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)