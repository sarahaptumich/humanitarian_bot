# Additional imports for debugging
import numpy as np

# Updated Streamlit code

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
            files = fs.ls(f"{bucket_name}/chroma_db_persist/")
            
            for file_path in files:
                local_file_path = os.path.join(chroma_db_path, os.path.basename(file_path))
                fs.get(file_path, local_file_path)
        except Exception as e:
            st.error(f"Error accessing or downloading from bucket: {e}")

        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=chroma_db_path,
            collection_name="nonprofit_reports"
        )

        # Debug: Display all documents in the database
        st.subheader("Chroma Database Debugging")
        all_docs = vectorstore._collection.get(include=['metadatas', 'documents'])
        st.write(f"Total documents in Chroma collection: {len(all_docs['documents'])}")
        for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas']), start=1):
            st.write(f"**Document {i}:**")
            st.write(f"**Metadata:** {metadata}")
            st.write(f"**Content Preview:** {doc[:500]}...")  # Display first 500 characters of content

        # Debug: Query embedding
        query_embedding = embeddings.embed_query(query)
        st.write("Query Embedding Vector (first 10 values):", query_embedding[:10])

        # Debug: Calculate similarity scores for all documents
        st.subheader("Document Similarity Scores")
        doc_embeddings = vectorstore._collection.get_embeddings()
        similarity_scores = [
            np.dot(query_embedding, doc_embedding) for doc_embedding in doc_embeddings
        ]
        sorted_scores = sorted(
            enumerate(similarity_scores),
            key=lambda x: x[1],
            reverse=True
        )
        for idx, score in sorted_scores:
            metadata = all_docs['metadatas'][idx]
            st.write(f"Document {idx + 1}: Score={score:.4f}, Metadata={metadata}")

        # Retrieve relevant documents directly using the user query
        with st.spinner("Retrieving relevant documents..."):
            docs = vectorstore.similarity_search(query, k=max_docs)
        
        # Debug: Retrieved documents and their scores
        st.subheader("Retrieved Documents with Scores")
        for i, doc in enumerate(docs, start=1):
            st.write(f"**Document {i} Metadata:** {doc.metadata}")
            st.write(f"**Content Preview:** {doc.page_content[:500]}...")  # Display first 500 characters

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
        
            # Link to open the PDF if available and embed a viewer
            if 'file_path' in metadata:
                file_name = metadata['file_path'].split('/')[-1]
                public_url = f"https://storage.googleapis.com/{bucket_name}/analysis/{file_name}"  # Adjust as needed
            
                # Display a clickable link to open the PDF in a new tab
                st.markdown(
                    f"[**Open PDF**]({public_url})",
                    unsafe_allow_html=True,
                )
            
            # Expanders to show page content and metadata JSON
            with st.expander("Show Page Content"):
                st.write(doc.page_content)
            with st.expander("Show Metadata JSON"):
                st.json(metadata)


