import streamlit as st
import PyPDF2 
from sentence_transformers import SentenceTransformer 
import faiss
import numpy as np
import textwrap
from test_inference import chatbot
import time

st.title("ChatBot")
query=st.chat_input("Ask a question about the PDF")
# File uploader
with st.sidebar:
   uploaded_file = st.file_uploader("Upload a PDF", type=["pdf","txt"])
   huggingface_api_key = st.text_input("Huggingface OpenAI API Key", key="chatbot_api_key", type="password")
   "[Get an Huggingface API key](https://huggingface.co/settings/tokens/new?tokenType=read)"
    
if "history" not in st.session_state:
    st.session_state.history = [] 
for role,message in st.session_state.history:
    st.chat_message(role).write(message)    

if uploaded_file is not None:
    # Read PDF 
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
        a=textwrap.wrap(text,width=1000) 
    else:    
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() 
        a=textwrap.wrap(text,width=1000) 
    
else:
    st.warning("Please upload a PDF or text file to start the chat.")    
if query:
    
    st.chat_message("user").write(query)
    st.session_state.history.append(("user", query))

    embed_model=SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(text)
    dim = embeddings.shape[0] 
    index = faiss.IndexFlatL2(dim)
    embeddings = embeddings.reshape(1, -1)  
    index.add((embeddings))  
    
 
    query_embedding=embed_model.encode([query])
    distances,indices=index.search(np.array(query_embedding),k=4)
    retrieved_texts=[a[i] for i in indices[0]]
    context=" ".join(retrieved_texts)
    prompt=f"You are a helpful assistant for our company's knowledge base. Use ONLY the context provided below to answer the question. If the answer is not in the context, you must respond with the exact phrase: 'I could not find an answer in the knowledge base.' Do not try to make up an answer.Context: {context}Question: {query}"



    with st.chat_message("assistant"): 
        message_placeholder = st.empty()
        full_response = ""
        if (huggingface_api_key) :
            try:
                assistant_response = chatbot(prompt, huggingface_api_key)
            except Exception as e:
                assistant_response = f"Error: {str(e)}" 
        else:
            assistant_response = "Please enter your Huggingface API key in the sidebar to get a response."
            # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
                # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)






    

    st.session_state.history.append(("assistant", full_response))