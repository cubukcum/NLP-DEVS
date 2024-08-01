import pandas as pd
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# Root route
@app.route('/')
def home():
    return "Welcome to the Product Analyzer API!"

# Serve a favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(directory='static', path='favicon.ico', mimetype='image/vnd.microsoft.icon')

# Initialize model components outside the route for efficiency
MODEL_NAME = "llama3"

# Define the ProductAnalyzer class
class ProductAnalyzer:
    def __init__(self, csv_file_path, model_name=MODEL_NAME):
        self.csv_file_path = csv_file_path
        self.loader = CSVLoader(file_path=csv_file_path, encoding='utf-8')
        self.data = self.loader.load()
        self.model = Ollama(model=model_name)
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vector_store = FAISS.from_documents(self.data, embedding=self.embeddings)
        self.retriever = self.vector_store.as_retriever()
        
        # Define the prompt template
        self.template = """
        You are a product analyzer, analyze the whole comments on the products and give the overall evaluation of the product.

        Context: {context}

        Question: {question}

        Answer:
        """
        self.prompt = PromptTemplate.from_template(self.template)
        
        # Create the processing chain
        self.chain = (
            {
                "context": itemgetter("question") | self.retriever,
                "question": itemgetter("question"),
            }
            | self.prompt
            | self.model
        )
    
    def analyze_product(self, question):
        response = self.chain.invoke({"question": question})
        return response

@app.route('/analyze_product', methods=['POST'])
def analyze_product():
    # Get request data
    try:
        data = request.get_json()
        print(data)
        question = data.get("question", "What is the overall evaluation of the product?")
        csv_file_path = data.get("csv_file_path", "reviews1.csv")
        
        # Create an analyzer instance
        analyzer = ProductAnalyzer(csv_file_path)
        
        # Get the response from the analyzer
        response = analyzer.analyze_product(question)
        print(response)
        return jsonify({"response": response})
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == '__main__':
    start_app()
