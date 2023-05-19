from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import json
import re
from langchain.llms import OpenAI
from langchain import LLMChain, PromptTemplate, HuggingFaceHub
import os
import tiktoken
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

pinecone.init(api_key=os.environ.get("pinecone_key"), environment="us-west4-gcp")
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("openai_key"))
index_name = "langchain-01"


def num_tokens(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens_count = len(encoding.encode(string))
    print(tokens_count)


# num_tokens("what is pi?", "cl100k_base")


def fetch_page(url):
    url = url
    headers = {"User-Agent": "Real Person"}
    request = Request(url, headers=headers or {})
    page = urlopen(request, timeout=10)
    soup = BeautifulSoup(page, "html.parser")
    # Find all the HTML elements that contain the text that we want to extract
    text_elements = soup.find_all(["h1", "p", "div", "pre"])

    # Extract the text from each element in the order they appear on the webpage
    text_list = []
    for element in text_elements:
        text = element.get_text(strip=False)
        if text:
            text = re.sub(r"[\n]", " ", text)
            text_list.append(text)

    # split text into chunks that can be embedded and indexed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_chunks = text_splitter.split_text(str(text_list))

    doc_search = Pinecone.from_texts(
        [chunk for chunk in text_chunks],
        embeddings,
        index_name=index_name,
    )

    query = "What is the default backend for Ivy?"
    docs = doc_search.similarity_search(query, include_metadata=True)

    print(docs)


def retreive_docs(query):
    docs_search = Pinecone.from_existing_index(index_name, embeddings)
    docs = docs_search.similarity_search(query, include_metadata=True)
    llm = OpenAI(temperature=0.0, openai_api_key=os.environ.get("openai_key"))
    chain = load_qa_chain(llm, chain_type="stuff")
    res = chain.run(input_documents=docs, question=query)
    print(res)


q = "What is the default backend for Ivy?"
retreive_docs(q)


# fetch_page("https://lets-unify.ai/docs/ivy/overview/design/building_blocks.html")


def gpt_llm_call():
    template = """Question: {question}

    Answer: """
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # user question
    question = "Which NFL team won the Super Bowl in the 2010 season?"

    chatGPT = OpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=os.environ.get("openai_key")
    )

    llm_chain = LLMChain(prompt=prompt, llm=chatGPT)
    print(llm_chain.run(question))


def huggingface_llm_call():
    template = """Question: {question}

    Answer: """
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # user question
    question = "Which NFL team won the Super Bowl in the 2010 season?"

    hub_llm = HuggingFaceHub(
        repo_id="google/flan-t5-xl",
        model_kwargs={"temperature": 1e-10},
        huggingfacehub_api_token=os.environ.get("huggingface_key"),
    )

    llm_chain = LLMChain(prompt=prompt, llm=hub_llm)
    print(llm_chain.run(question))


# huggingface_llm_call()
