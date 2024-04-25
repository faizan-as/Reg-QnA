import streamlit as st 
import os
import base64
# import time
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
# import torch 
# from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from streamlit_chat import message
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_chroma import Chroma
# from langchain_community.embeddings.sentence_transformer import (
#     SentenceTransformerEmbeddings,
# )
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv() 

st.set_page_config(layout="wide")

persist_directory = "db"

@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002")

    #create vector store here
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    #db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)

    # db.persist()
    db=None 

@st.cache_resource
def az_llm():
    llm = AzureChatOpenAI(temperature=0, deployment_name="gpt-35-turbo")
    return llm

@st.cache_resource
def qa_llm():
    llm = az_llm()
    #embeddings = SentenceTransformerEmbeddings(model_name="text-embedding-ada-002")
    embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002")

    db = Chroma(persist_directory="db", embedding_function = embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data
#function to display the PDF of a given file 
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))

def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>RegSmart - Regulatory Intelligence Chatbot 📄💬 </h1>", unsafe_allow_html=True)
    
    col1, col2= st.columns([1,2])
    with col1:
        uploaded_file = st.file_uploader("", type=["pdf"])

        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": get_file_size(uploaded_file)
            }
            filepath = "docs/"+uploaded_file.name
            with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

            with st.spinner('Embeddings are in process...'):
                ingested_data = data_ingestion()
            st.success('Embeddings are created successfully!')
            #st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)
        
        pre_definedQuestions()

    with col2:
        qa_chain = qa_llm()

        if qa_chain is not None:
            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = []
            if "past" not in st.session_state:
                st.session_state["past"] = []
            
            user_question = st.text_input("Search in Regulatory Documents:", value=st.session_state.get("user_question", ""))
            st.session_state.user_question = user_question
        
            # Add a Search button
            if st.button("Ask RegSmart"):
                # Search the database for a response based on user input and update session state
                if user_question:
                    # print(user_question)
                    answer = process_answer({'query': user_question})
                    st.session_state["past"].append(user_question)
                    response = answer
                    st.session_state["generated"].append(response)

                # Display conversation history using Streamlit messages
                if st.session_state["generated"]:
                    display_conversation(st.session_state)

def pre_definedQuestions():
    # Predefined Questions
    question_1 = "what are key elements of impactful digital therapeutics solutions"
    question_2 = "What impact has the COVID-19 pandemic likely had on digital health value pools?"
    question_3 = "what is the role of Digital therapeutics in chronic disease management?"
    question_4 = "List the critical deficiencies around pharmacovigilance inspections?"
    
    st.write("You can ask questions like:")
    
    # Four buttons with predefined questions
    if st.button(question_1):
        st.session_state.user_question = question_1
    elif st.button(question_2):
        st.session_state.user_question = question_2
    elif st.button(question_3):
        st.session_state.user_question = question_3
    elif st.button(question_4):
        st.session_state.user_question = question_4


if __name__ == "__main__":
    main()


