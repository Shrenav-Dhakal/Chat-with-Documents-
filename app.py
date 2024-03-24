import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdfs):
    texts=""
    for pdf in pdfs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            texts+= page.extract_text()
    return texts

def generate_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def generate_vectors(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectors = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectors.save_local("faiss_index")

def conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, call_me=False, name=None, phone=None, email=None):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings=embeddings,  allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = conversation_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    if "call me" in user_question.lower():
        st.write("Reply: ", response["output_text"])
        st.write("Please provide your contact details:")
        name = st.text_input("Your Name:")
        phone = st.text_input("Your Phone Number:")
        email = st.text_input("Your Email Address:")
        st.write("Thank you! We will get in touch with you shortly.")
    else:
        st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chatbot")
    st.header("Chat with PDF Documents")

    user_question = st.text_input("Ask a Question from the PDF Files or ask us to call user")
    name, phone, email = None, None, None

    if "call me" in user_question.lower():
        user_input(user_question, True, name, phone, email)
    elif user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdfs = st.file_uploader("Upload your PDF Files and Click on the Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdfs)
                chunks = generate_chunks(raw_text)
                generate_vectors(chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
