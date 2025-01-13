import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
import os
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import gcsfs
from google.cloud import storage

# Configure Streamlit page
st.set_page_config(page_title="Humanitarian ChatBot", page_icon=":earth_americas:")
st.title("Humanitarian ChatBot")

# Set GOOGLE_APPLICATION_CREDENTIALS dynamically
try:
    service_account_json = st.secrets["general"]["GOOGLE_APPLICATION_CREDENTIALS_JSON"]

    # Write the JSON key to a temporary file
    temp_credentials_path = "/tmp/service_account_key.json"
    with open(temp_credentials_path, "w") as f:
        f.write(service_account_json)

    # Set the environment variable for Google Cloud SDK
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_credentials_path
except KeyError:
    st.error("The GOOGLE_APPLICATION_CREDENTIALS_JSON key is missing in your Streamlit secrets.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# Debug GCS access
client = storage.Client()
try:
    buckets = list(client.list_buckets())
except Exception as e:
    st.error(f"Error accessing GCS: {e}")

# Sidebar settings
st.sidebar.image("humanitarian_bot.jpeg")
st.sidebar.header("Settings")

# Model selection and configuration
model_options = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
selected_model = st.sidebar.selectbox("Select LLM Model for Answer", model_options, index=0)
temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.5, 0.05)
max_docs = st.sidebar.slider("Maximum number of documents", 1, 10, 3, 1)

# User input
st.write("Ask a question about the nonprofit reports:")
query = st.text_input("Your question", "")
submit = st.button("Submit")

# Cache ChromaDB Initialization
@st.cache_data(persist="disk")
def initialize_chroma(embedding_model_name, chroma_db_path):
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    # Initialize vector store
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=chroma_db_path,
        collection_name="nonprofit_reports"
    )
    return vectorstore, embeddings

if submit and query.strip():
    google_api_key = st.secrets["general"].get("GOOGLE_API_KEY")
    
    if not google_api_key:
        st.error("Google API key not found. Please set the GOOGLE_API_KEY in your secrets.")
    else:
        # System prompt for the final answer
        system_prompt = (
            "You are a Q&A assistant dedicated to providing accurate, up-to-date information "
            "from ReliefWeb, a humanitarian platform managed by OCHA. Use the provided context documents "
            "to answer the user’s question. If you cannot find the answer or are not sure, say that you do not know. "
            "Keep your answer to ten sentences maximum, be clear and concise. Always end by inviting the user to ask more!"
        )

        # Access GCS bucket and download files locally
        fs = gcsfs.GCSFileSystem()
        bucket_name = "humanitarian_bucket"
        chroma_db_path = "/tmp/chroma_db_persist"  # Local directory for Chroma persistence

        # Ensure local directory exists
        os.makedirs(chroma_db_path, exist_ok=True)

        try:
            # List files in the GCS bucket and download to local directory
            files = fs.find(f"{bucket_name}/chroma_db_persist/")
            for file_path in files:
                local_file_path = os.path.join(chroma_db_path, os.path.basename(file_path))
                fs.get(file_path, local_file_path)
            st.success(f"Downloaded {len(files)} files from GCS.")
        except Exception as e:
            st.error(f"Error accessing or downloading from bucket: {e}")

        # Initialize Chroma database
        vectorstore, embeddings = initialize_chroma("BAAI/bge-base-en-v1.5", chroma_db_path)
        st.success("Chroma database initialized.")

        ##### Hypothetical Question Generation #####
        st.subheader("Hypothetical Question Generation")

        # Define the hypothetical question prompt
        hypo_system_prompt = """
        You are an expert assistant for humanitarian information from ReliefWeb.
        Given a user's query, rewrite or refine the query into a hypothetical question
        that is more specific and informative, while retaining the original intent.
        """
        hypo_prompt = ChatPromptTemplate.from_messages([
            ("system", hypo_system_prompt),
            ("human", "{question}")
        ])
        hypo_llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0.5, api_key=google_api_key)
        hypo_chain = hypo_prompt | hypo_llm | StrOutputParser()

        # Generate the hypothetical question
        with st.spinner("Generating hypothetical question..."):
            hypothetical_question = hypo_chain.invoke({"question": query})
            st.write("**Hypothetical Question:**", hypothetical_question)

        # Use the hypothetical question for retrieval
        query = hypothetical_question
        ##### End Hypothetical Question Generation #####

        # Display the total number of documents in the Chroma database
        st.subheader("Chroma Database Document Count")
        document_count = len(vectorstore._collection.get()['documents'])
        st.write(f"Total documents in Chroma database: {document_count}")

        # Retrieve relevant documents directly using the hypothetical question
        with st.spinner("Retrieving relevant documents..."):
            docs = vectorstore.similarity_search(query, k=max_docs)

        # Calculate similarity scores for retrieved documents only
        st.subheader("Similarity Scores of Retrieved Documents")
        query_embedding = embeddings.embed_query(query)  # Compute the query embedding
        retrieved_scores = []

        # Iterate over the retrieved documents and compute similarity scores
        for doc in docs:
            doc_embedding = embeddings.embed_query(doc.page_content)  # Recompute document embedding
            similarity = np.dot(query_embedding, doc_embedding)  # Calculate similarity
            retrieved_scores.append((doc, similarity))

        # Display the retrieved documents and their similarity scores
        for i, (doc, score) in enumerate(retrieved_scores, start=1):
            st.write(f"**Document {i}:**")
            st.write(f"**Similarity Score:** {score:.4f}")
            st.write(f"**Metadata:** {doc.metadata}")
            st.write(f"**Content Preview:** {doc.page_content[:500]}...")  # Show the first 500 characters

        # Prepare the final chain using the selected model
        final_llm = ChatGoogleGenerativeAI(model=selected_model, temperature=temperature, api_key=google_api_key)
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        # Create the chain to combine documents and generate the final answer
        combine_docs_chain = create_stuff_documents_chain(
            llm=final_llm,
            prompt=retrieval_qa_chat_prompt
        )

        # Generate the final answer by invoking the chain with the context and user input
        with st.spinner("Creating final answer..."):
            response = combine_docs_chain.invoke({
                "context": docs,
                "input": f"{system_prompt}\n\nUser question: {query}"
            })

        # Display the final answer
        st.subheader("Agent Response")
        st.write(response)
