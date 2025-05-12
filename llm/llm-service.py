from flask import Flask, request, jsonify
import requests
import os
import google.generativeai as genai

from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Add this line after creating your Flask app

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# FAISS service URL
FAISS_SERVICE_URL = os.environ.get("FAISS_SERVICE_URL", "http://faiss-service:5001/query")
print(f"FAISS Service URL: {FAISS_SERVICE_URL}")

@app.route("/rag", methods=["POST"])
def rag():
    """Answer questions using RAG: FAISS + Gemini"""
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Retrieve context from FAISS
    response = requests.post(FAISS_SERVICE_URL, json={"question": question})
    context = " ".join(response.json().get("results", []))

    # Prompt Gemini
    prompt = f"""
    You are a financial expert.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    answer = model.generate_content(prompt).text

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)