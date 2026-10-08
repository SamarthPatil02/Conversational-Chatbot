import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# Hugging Face Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Streamlit UI
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content")

# Groq API Key
api_key = st.text_input("Enter your Groq API Key:", type="password")

if api_key:

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

    # Session ID
    session_id = st.text_input(
        "Session ID",
        value="default_session"
    )

    # Session Store
    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:

        documents = []

        # Load all PDF
        for uploaded_file in uploaded_files:

            temp_pdf = f"./temp_{os.path.basename(uploaded_file.name)}"

            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()

            documents.extend(docs)

            os.remove(temp_pdf)  # delete the temporary copy after loading

        # Split Documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500
        )

        splits = text_splitter.split_documents(documents)

        # Vector Store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding
        )

        retriever = vectorstore.as_retriever()

        # Contextualization Prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question that can be understood "
            "without the chat history. Do not answer the question."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )

        # QA Prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following retrieved context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use at most three sentences.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        # Chat History Function
        def get_session_history(
            session_id: str
        ) -> BaseChatMessageHistory:

            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()

            return st.session_state.store[session_id]

        # Conversational Chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User Question
        user_input = st.text_input(
            "Ask a question about the uploaded PDFs:"
        )

        if user_input:

            session_history = get_session_history(session_id)

            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {
                        "session_id": session_id
                    }
                }
            )

            st.subheader("Assistant Response")
            st.write(response["answer"])

            st.subheader("Chat History")
            st.write(session_history.messages)

else:
    st.warning("Please enter your Groq API Key.")
