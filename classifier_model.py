import pandas as pd
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from flask import Flask, request, jsonify, send_from_directory
from enum import Enum
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import instructor

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

class ProductAnalyzer:
    def _init_(self, csv_file_path, model_name=MODEL_NAME):
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


class CustomerSentiment(str, Enum):
    ANGRY = "KIZGIN"
    FRUSTRATED = "ÖFKELİ"
    NEUTRAL = "NÖTR"
    SATISFIED = "MEMNUN"


class ReviewCategory(str, Enum):
    PRODUCT = "product"
    SERVICE = "service"
    DELIVERY = "delivery"
    CUSTOMER_SUPPORT = "customer_support"
    PRICE = "price"
    OTHER = "other"


class ReviewClassification(BaseModel):
    sentiment: CustomerSentiment
    review_details: List[str] = Field(description="Key details extracted from the review from the list :etc.")
    confidence: float = Field(ge=0, le=1, description="Confidence score for the classification")
    summary: List[str] = Field(description="Summary of the review")
    suggestions: str = Field(description="Detailed suggestions for the customers who wants to buy this product")


# enables response_model in create call
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="sk-proj-ZkESFEWR9q2pi23yJy6MT3BlbkFJmSKLsoAWBRhSkdi15w3F",
    ),
    mode=instructor.Mode.JSON,
)


def classify_review(review: str) -> ReviewClassification:
    response = client.chat.completions.create(
        model="llama3",
        response_model=ReviewClassification,
        temperature=0.2,
        max_retries=3,
        messages=[
            {
                "role": "system",
                "content": "Analyze the following product review and extract the requested information."
            },
            {"role": "user", "content": review}
        ]
    )
    return response


@app.route('/analyze_product', methods=['POST'])
def analyze_product():
    try:
        data = request.get_json()
        question = data.get("question", "What is the overall evaluation of the product?")
        csv_file_path = data.get("csv_file_path", "reviews1.csv")

        # Create an analyzer instance
        analyzer = ProductAnalyzer(csv_file_path)

        # Get the response from the analyzer
        response = analyzer.analyze_product(question)
        return jsonify({"response": response})
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/classify_review', methods=['POST'])
def classify_review_endpoint():
    try:
        data = request.get_json()
        review = data.get("review", "")

        if not review:
            return jsonify({"error": "Review text is required"}), 400

        # Get the response from the classification function
        response = classify_review(review)
        return jsonify(response.dict())
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == '_main_':
    start_app()