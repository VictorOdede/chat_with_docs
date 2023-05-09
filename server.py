from flask import Flask, request, jsonify
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

pinecone.init(api_key=os.environ.get("pinecone_key"), environment="us-west4-gcp")
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("openai_key"))
index_name = "langchain-01"

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello, World! Serving this app from here"


@app.route("/chat/gpt3")
def chat_gpt():
    def retreive_docs(query):
        docs_search = Pinecone.from_existing_index(index_name, embeddings)
        docs = docs_search.similarity_search(query, include_metadata=True)
        llm = OpenAI(temperature=0.0, openai_api_key=os.environ.get("openai_key"))
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
    llm = OpenAI(temperature=0.0, openai_api_key=os.environ.get("openai_key"))
    chain = load_qa_chain(llm, chain_type="stuff")
    res = chain.run(input_documents=docs, question=query)
    return {"response": res}
