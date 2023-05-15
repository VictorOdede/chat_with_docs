from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import os
import io
import PyPDF2
import docx
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

pinecone.init(api_key=os.environ.get("pinecone_key"), environment="us-west4-gcp")
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("openai_key"))
index_name = "langchain-01"

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/")
def hello_world():
    return "Server is running 🚀🚀"


@app.route("/upload", methods=["POST"])
@cross_origin()
def upload_file():
    # Extract text from pdf/word/google-doc files

    if "document" not in request.files:
        return "No file uploaded", 400

    file = request.files["document"]
    file_name = secure_filename(file.filename)
    file_ext = file_name.split(".")[-1].lower()
    print(file_name)
    text = ""

    if file_ext == "pdf":
        pdf_reader = PyPDF2.PdfFileReader(io.BytesIO(file.read()))
        for page_num in range(pdf_reader.getNumPages()):
            text += pdf_reader.getPage(page_num).extractText()
    elif request.files["type"] in ["doc", "docx"]:
        doc = docx.Document(io.BytesIO(file.read()))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        return "Only PDF, Word and Google Doc files are allowed", 400

    print(text)

    # Create new Pinecone index
    # Store embeddings in Pinecone
    # Check user table in DB for existing Index
    # Delete old index from Pinecone
    # Replace old index with new index
    return "Document uploaded", 200


@app.route("/chat/gpt3")
def chat_gpt():
    def retreive_docs(query):
        docs_search = Pinecone.from_existing_index(index_name, embeddings)
        docs = docs_search.similarity_search(query, include_metadata=True)
        llm = OpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            openai_api_key=os.environ.get("openai_key"),
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        res = chain.run(input_documents=docs, question=query)
        return res

    ret_val = retreive_docs("What is the default backend for Ivy?")
    return ret_val


@app.route("/chat/test", methods=["POST"])
def start_conv():
    data = request.json
    print(data["question"])
    query = data["question"]

    vectorstore = Pinecone.from_existing_index(index_name, embeddings)
    docs = vectorstore.similarity_search(query, include_metadata=True)
    llm = OpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        openai_api_key=os.environ.get("openai_key"),
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    res = chain.run(input_documents=docs, question=query)
    return {"response": res}


if __name__ == "__main__":
    app.run(debug=True)
