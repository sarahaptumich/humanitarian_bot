import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import gcsfs

# Configure Streamlit page
st.set_page_config(page_title="Humanitarian ChatBot", page_icon=":earth_americas:")
st.title("Humanitarian ChatBot")

# Sidebar settings
st.sidebar.image("humanitarian_bot.jpeg")
st.sidebar.header("Settings")

# Model selection and configuration
model_options = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
selected_model = st.sidebar.selectbox("Select Final LLM Model for Answer", model_options, index=0)
temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.5, 0.05)
max_docs = st.sidebar.slider("Maximum number of documents", 1, 10, 3, 1)

# User input
st.write("Ask a question about the nonprofit reports:")
query = st.text_input("Your question", "")
submit = st.button("Submit")

if submit and query.strip():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
    else:
        # System prompt for the final answer
        system_prompt = (
            "You are a Q&A assistant dedicated to providing accurate, up-to-date information "
            "from ReliefWeb, a humanitarian platform managed by OCHA. Use the provided context documents "
            "to answer the user’s question. If you cannot find the answer or are not sure, say that you do not know. "
            "Keep your answer to ten sentences maximum, be clear and concise. Always end by inviting the user to ask more!"
        )

        # Access GCS bucket using gcsfs
        fs = gcsfs.GCSFileSystem()
        bucket_name = "humanitarian_bucket"

        # Mount the chroma_db_persist directory
        chroma_db_path = f"{bucket_name}/chroma_db_persist"
        analysis_path = f"{bucket_name}/analysis"

        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=chroma_db_path,
            collection_name="nonprofit_reports",
            filesystem=fs
        )

        # Hypothetical document generation prompt
        hypo_system_prompt = """You are an expert about humanitarian information from ReliefWeb.
Answer the user's question as best you can, as though you were writing a reference document.
Be factual and topic-focused, even if you have to guess based on general knowledge."""

        # Create the prompt template for hypothetical document generation
        hypo_prompt = ChatPromptTemplate.from_messages([
            ("system", hypo_system_prompt),
            ("human", "{question}")
        ])

        # Initialize the LLM for hypothetical document generation
        hypo_llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0)
        qa_no_context = hypo_prompt | hypo_llm | StrOutputParser()

        # Generate the hypothetical document
        with st.spinner("Optimizing query..."):
            hypothetical_doc = qa_no_context.invoke({"question": query})

        # Embed the hypothetical document
        hypothetical_embedding = embeddings.embed_query(hypothetical_doc)

        # Retrieve relevant documents using the hypothetical embedding
        with st.spinner("Retrieving relevant documents..."):
            docs = vectorstore.similarity_search_by_vector(hypothetical_embedding, k=max_docs)

        # Prepare the final chain using the selected model
        final_llm = ChatGoogleGenerativeAI(model=selected_model, temperature=temperature)
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

        # Display the retrieved documents with metadata
        st.subheader("Retrieved Documents")
        for i, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            title = metadata.get('title', 'No title available')
            page_label = metadata.get('page_label', 'N/A')
            created_date = metadata.get('date.created', 'N/A')
            country = metadata.get('country', 'N/A')
            
            st.write(f"**Document {i}: {title}**")
            st.write(f"**Page #:** {page_label}")
            st.write(f"**Date Created:** {created_date}")
            st.write(f"**Country:** {country}")

            # Add a link to open the PDF if available
            if 'file_path' in metadata:
                file_name = metadata['file_path'].split('/')[-1]
                file_url = f"http://localhost:8502/{file_name}"
                st.markdown(
                    f"<a href='{file_url}' target='_blank'>Open PDF</a>",
                    unsafe_allow_html=True
                )

            # Expanders to show page content and metadata JSON
            with st.expander("Show Page Content"):
                st.write(doc.page_content)
            with st.expander("Show Metadata JSON"):
                st.json(metadata)
