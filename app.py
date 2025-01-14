import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure Streamlit page
st.set_page_config(page_title="Humanitarian ChatBot", page_icon=":earth_americas:")
st.title("Humanitarian ChatBot")

# Sidebar settings
# Sidebar settings
st.sidebar.image("humanitarian_bot.jpeg", use_container_width=True)
st.sidebar.header("Settings")
# Model selection and configuration
model_options = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
selected_model = st.sidebar.selectbox("Select LLM Model for Answer", model_options, index=0)
temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.5, 0.05)
top_k = st.sidebar.slider("Number of documents to retrieve", 1, 10, 3, 1)

# User input
query = st.text_input("Ask a question about the nonprofit reports:")
submit = st.button("Submit")


# Function to generate a hypothetical question
def generate_hypothetical_question(query, model, temperature, api_key):
    prompt_template = """
    You are an expert assistant for humanitarian information from ReliefWeb.
    Rewrite or refine the user's query into a more specific hypothetical question 
    that is clear, informative, and retains the original intent.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("human", "{question}")
    ])
    llm = ChatGoogleGenerativeAI(model=model, temperature=0.60, api_key=api_key)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": query})

if submit and query.strip():
    google_api_key = st.secrets["general"].get("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Google API key not found. Please set it in Streamlit secrets.")
    else:
        with st.spinner("Generating hypothetical question..."):
            hypothetical_question = generate_hypothetical_question(query, selected_model, temperature, google_api_key)
            st.subheader("Step 1: Hypothetical Question")
            st.write(f"**Refined Query:** {hypothetical_question}")