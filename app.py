import streamlit as st
# import pypdf as pdfReader
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from PyPDF2 import PdfReader


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    """
    Extracts text content from one or more PDF documents.

    Args:
        pdf_docs (list): A list of file objects representing PDF documents.

    Returns:
        str: A string containing the concatenated text content of all pages
             from all provided PDF documents.

    This function iterates through each PDF document in the input list,
    reads all pages, and extracts the text content. The extracted text
    from all pages and all documents is combined into a single string.
    """
     
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")


def get_conversation_chain():
    prompt_template =  """
    Answer the question as detailed as porssible from the provided context, make sure to provide all the details
    if the answer is not in the context provided then say, 'The answer is not in the context provided', dont provide a wrong answer. 
    Also provide the page number in the pdf where the answer is found.
    Provide the answer in the same language of the question.

    Context: \n{context} \n
    Question: \n{question} \n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def handle_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain()

    response = chain({
        "input_documents": docs,
        "question": user_question}, return_only_outputs=True)

    print(response)
    st.write(response["output_text"])


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs using Gemini Pro")

    user_question = st.text_input("Ask a question from the PDF files:")


    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text = get_chunks(raw_text)
                get_vector_store(text)
                st.success("Done! You can ask questions from the PDFs now.")


if __name__ == "__main__":
    main()