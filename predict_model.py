from flask import Flask, request, jsonify
import pandas as pd
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from operator import itemgetter

# Initialize Flask app
app = Flask(__name__)

# Initialize components
MODEL = "llama3"
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

# Path to the CSV file
csv_file_path = "reviews1.csv"  # Update with your CSV file path

# Load the CSV data
loader = CSVLoader(file_path=csv_file_path, encoding='utf-8')
data = loader.load()

# Initialize the vector store and retriever
vector_store = FAISS.from_documents(data, embedding=embeddings)
retriever = vector_store.as_retriever()

# Define the prompt template
template = """
You are a product analyzer, analyze the whole comments on the products and give the overall evaluation of the product.

Context: {context}

Question: {question}

Answer :
"""

prompt = PromptTemplate.from_template(template)

# Create the chain
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
)

@app.route('/evaluate', methods=['POST'])
def evaluate_product():
    data = request.json
    question = data.get('question', '')

    # Check if a question is provided
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Generate response
    response = chain.invoke({"question": question})
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
