import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText

# Load environment variables from a .env file
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Set your Gemini API key


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Replace if using a different model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save the vector store for later use
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Answer the question in as detailed manner as possible from the provided context, make sure to provide all the details, if the answer is not in the provided
    context then just say, "answer is not available in the context", dont provide the wrong answer\n\n
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Replace if using a different model

    try:
        new_db = FAISS.load_local("faiss_index", embeddings)  # Load the saved vector store
    except FileNotFoundError:  # Handle the case where the vector store doesn't exist yet
        st.error("Please upload PDFs and process them first.")
        return

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    st.write("Reply:", response["output_text"])

    # Send email if response is long
    if len(response["output_text"]) > 1000:
        send_email(response["output_text"])


def send_email(response_text):
    sender_email = "ritikadobhal710@gmail.com"
    receiver_email = "2023.ritika.dobhal@ves.ac.in"
    subject = "PDF Chatbot Response"

    # Create email message
    message = MIMEText(response_text)
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Set up SMTP server
    smtp_server = "smtp.gmail.com"  # Replace with your SMTP server
    port = 587
    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls()
        server.login(sender_email, "sbpk ovot ccck cwsy")  # Replace with your password
        server.sendmail(sender_email, receiver_email, message.as_string())
    print("Email sent successfully!")


def main():
    st.set_page_config(page_title="Chat PDF", page_icon=":file_pdf:")  # Set title and icon

    # Use markdown for button styling
    st.markdown ("""
        <style>
        .custom-button {
            background-color: #9b59b6;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .custom-button:hover {
            background-color: #8e44ad;
        }
        </style>
        <button class="custom-button" onclick="window.location.reload();">Submit & Process</button>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Upload PDF Files:")
        pdf_docs = st.file_uploader("Select PDF files", accept_multiple_files=True)

        if st.button ("Submit & Process"):
            with st.spinner ("Processing..."):
                raw_text = get_pdf_text (pdf_docs)
                text_chunks = get_text_chunks (raw_text)
                get_vector_store (text_chunks)
            st.success ("Done!")

    with col2:
        st.subheader("Ask Your Question:")
        user_question = st.text_input ("Enter your question", key="question_input")

        if user_question:
            user_input(user_question)

if __name__ == "__main__":
    main()