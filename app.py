import streamlit as st
import PyPDF2 
from sentence_transformers import SentenceTransformer 
import faiss
import numpy as np
import textwrap
from transformers import pipeline


st.title("ChatBot")
query=st.chat_input("Ask a question about the PDF")
# File uploader
with st.sidebar:
   uploaded_file = st.file_uploader("Upload a PDF", type=["pdf","txt"])
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
    
    qa_pipeline=pipeline("text2text-generation",model="google/flan-t5-small")
    query_embedding=embed_model.encode([query])
    distances,indices=index.search(np.array(query_embedding),k=4)
    retrieved_texts=[a[i] for i in indices[0]]
    context=" ".join(retrieved_texts)
    prompt=f"context: {context} \n\nQuestion: {query}\nAnswer"
    result=qa_pipeline(prompt,max_length=200,do_sample=False) 
    st.chat_message("assistant").markdown(result[0]['generated_text'])
    st.session_state.history.append(("assistant", result[0]['generated_text']))