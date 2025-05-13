import streamlit as st 
import os 

st.set_page_config(page_title="PDF Summarizer") 

st.title("PDF Summarizer") 
st.write("Summarize your pdf files using hte power of LLMs") 
st.divider() 
pdf = st.file_uploader("Upload your PDF", type="pdf")  
submit = st.button("Generate Summary") 



from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import openai
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from pypdf import PdfReader

def process_text(text): 
  text_splitter = CharacterTextSplitter( 
    separator="\n", 
    chunk_size=1000, 
    chunk_overlap=200, 
    length_function=len 
  )

  chunks = text_splitter.split_text(text)
  embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') 

  knowledgeBase = FAISS.from_texts(chunks, embeddings) 
  return knowledgeBase 

def summarizer(pdf): 
  response = ""
  pdf_reader = PdfReader(pdf) 
  text = "" 

  # Extract text from each page of the PDF 
  for page in pdf_reader.pages: 
    text += page.extract_text() or "" 
    knowledgeBase = process_text(text) 
    query = "Summarize the content of the uploaded PDF file in approximately 5-8 sentences." 

    # Load the question and answer chain
    if query: 
      docs = knowledgeBase.similarity_search(query) 
      OpenAIModel = "gpt-3.5-turbo-16k" 
      llm = ChatOpenAI(model=OpenAIModel, temperature=0.1) 
      chain = load_qa_chain(llm, chain_type='stuff') 

      #Run the above chain through ChatGPT model to get results 
      with get_openai_callback() as cost: 
        response = chain.run(input_documents=docs, question=query) 
        print(cost)  
  return response 

os.environ["OPENAI_API_KEY"] = "Your API KEY" 

# Call the `summarizer()` function when the `Generate Summary` button is clicked 
if submit: 
  response = summarizer(pdf) 

  # Display the returned summary 
  st.subheader("PDF Summary") 
  st.write(response) 
