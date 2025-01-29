import os
import streamlit as st
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

# Configure Streamlit page
st.set_page_config(page_title="Humanitarian ChatBot", page_icon=":earth_americas:")
st.title("Humanitarian ChatBot")

# Sidebar settings
st.sidebar.image("humanitarian_bot.jpeg")
st.sidebar.header("Settings")

# Model selection and configuration
model_options = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
selected_model = st.sidebar.selectbox("Select LLM Model for Answer", model_options, index=0)
temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.5, 0.05)
k = st.sidebar.slider("Number of Similar Documents (k)", 1, 10, 5, 1)

# User input
st.write("Ask a question about the nonprofit reports:")
query = st.text_input("Your question", "")
submit = st.button("Submit")

# Similarity API URL
SIMILARITY_API_URL= st.secrets["general"].get("SIMILARITY_API")

if submit and query.strip():
    google_api_key = st.secrets["general"].get("GOOGLE_API_KEY")

    if not google_api_key:
        st.error("Google API key not found. Please set the GOOGLE_API_KEY in your secrets.")
    else:
        ##### Call Similarity API #####
        st.subheader("Retrieving Similar Documents")
        with st.spinner("Finding similar documents..."):
            payload = {"text": query, "k": k}
            try:
                response = requests.post(SIMILARITY_API_URL, json=payload)
                response.raise_for_status()
                similar_docs = response.json().get("results", [])
            except requests.exceptions.RequestException as e:
                st.error(f"Error calling similarity API: {e}")
                st.stop()

    # Debug API Response
st.subheader("Debugging Similarity API Response")

with st.spinner("Fetching data from Similarity API..."):
    payload = {"text": query, "k": k}
    try:
        response = requests.post(SIMILARITY_API_URL, json=payload)
        response.raise_for_status()
        similar_docs = response.json().get("results", [])

        # Print full API response for debugging
        st.write("**Raw API Response:**", similar_docs)

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling similarity API: {e}")
        st.stop()


        ##### Display Retrieved Documents #####
        for i, doc in enumerate(similar_docs, start=1):
            st.write(f"**Document {i}:**")
            st.write(f"**Title:** {doc.get('title')}")
            st.write(f"**Content Preview:** {doc.get('content')[:500]}...")  # Show the first 500 characters

        ##### Generate Final Answer Using Gemini #####
        if similar_docs:
            st.subheader("Generating Final Answer")
            context = "\n\n".join([doc.get("content") for doc in similar_docs])
            system_prompt = (
                "You are a Q&A assistant dedicated to providing accurate, up-to-date information "
                "from ReliefWeb, a humanitarian platform managed by OCHA. Use the provided context documents "
                "to answer the user’s question. If you cannot find the answer or are not sure, say that you do not know. "
                "Keep your answer to ten sentences maximum, be clear and concise. Always end by inviting the user to ask more!"
            )

            retrieval_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser question: {query}"

            # Use Gemini to generate the response
            final_llm = ChatGoogleGenerativeAI(model=selected_model, temperature=temperature, api_key=google_api_key)
            with st.spinner("Creating final answer..."):
                try:
                    final_answer = final_llm.invoke(retrieval_prompt)
                    st.subheader("Agent Response")
                    st.write(final_answer)
                except Exception as e:
                    st.error(f"Error generating response with Gemini: {e}")
