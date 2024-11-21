import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPEN_API_KEY="your_secret_key"

st.header("My ChatBot")

with st.sidebar:
    st.title("My Documents")
    file=st.file_uploader("Upload a PDF file", type="pdf")
#extract the text
if file is not None:
    pdf_reader=PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        # st.write(text)

    text_splitter=RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    # st.write(chunks)

    #generating embeddings
    embeddings=OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)

    #vector storage
    vector_store=FAISS.from_texts(chunks, embeddings)

    #get question
    user_question=st.text_input("Type your question")

    if user_question:
        match=vector_store.similarity_search(user_question)
        st.write(match)

        #define the llm
        llm=ChatOpenAI(
            openai_api_key=OPEN_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        chain= load_qa_chain(llm, chain_type="stuff")
        response=chain.run(input_documents=match,question=user_question)
        st.write(response)
