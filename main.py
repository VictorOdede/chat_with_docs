from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import os
import io
import PyPDF2
import docx
import pinecone
from random_word import RandomWords
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


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
    elif file_ext in ["doc", "docx"]:
        doc = docx.Document(io.BytesIO(file.read()))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        return "Only PDF, Word and Google Doc files are allowed", 400

    print(text)

    # Initialize pinecone
    pinecone.init(api_key=os.environ.get("pinecone_key"), environment="us-west4-gcp")

    # Check user table in DB for existing Index
    all_indexes = pinecone.list_indexes()

    if len(all_indexes) > 0:
        # Delete old index from Pinecone
        pinecone.delete_index(all_indexes[0])

    # Create new Pinecone index
    r = RandomWords()
    index_name = r.get_random_word().lower()
    pinecone.create_index(index_name, dimension=1536)

    # specs = pinecone.describe_index(all_indexes[0])
    print(all_indexes)

    # Store embeddings in Pinecone
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("openai_key"))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_chunks = text_splitter.split_text(str(text))
    Pinecone.from_texts(
        [chunk for chunk in text_chunks],
        embeddings,
        index_name=index_name,
    )

    # Add index name to response
    return {"index_name": index_name}, 200


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
    # index_name = data["index"]
    # history = data["history"]

    # Initialize pinecone

    # return {"result": "Question received"}, 200

    # pass in chat history

    # set up db call to fetch index name
    try:
        pinecone.init(
            api_key=os.environ.get("pinecone_key"), environment="us-west4-gcp"
        )
        all_indexes = pinecone.list_indexes()
        index_name = all_indexes[0]
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("openai_key"))
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        docs = vector_store.similarity_search(query, include_metadata=True, k=3)
        llm = OpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            openai_api_key=os.environ.get("openai_key"),
        )

        chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
        res = chain.run(input_documents=docs, question=query)
        return {"result": res}
    except:
        raise Exception("Something went wrong!")
        # return "An error occurred", 500


if __name__ == "__main__":
    app.run(debug=True)
