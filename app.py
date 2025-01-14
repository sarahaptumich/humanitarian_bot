import streamlit as st

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