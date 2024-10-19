#!/usr/bin/env python
# coding: utf-8

# # AIM Hackathon: Sample code
# 19.10.2024

# In[3]:


import os
import requests
import PyPDF2
import tiktoken
import pandas as pd
import pickle
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings.base import OpenAIEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from Cryptodome.Cipher import AES

from typing import Optional, List, Union
from pydantic import BaseModel
from openai import OpenAI
import warnings

# Suppress only LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message="Importing FAISS from langchain.vectorstores is deprecated. Please replace deprecated imports")

# load openai key
if not load_dotenv():
    raise Exception('Error loading .env file. Make sure to place a valid OPEN_AI_KEY in the .env file.')


# In[4]:


REPORTS_SAVE_PATH = 'data/sample_reports'
DB_PATH = "data/db/sample.db"

# See https://openai.com/api/pricing/
MODEL = "gpt-4o"


# In[5]:


df = pd.read_json('data/reports.json')
df


# ## Download some reports

# In[6]:


# EXAMPLE: select apple reports
df_sample = df[df['dataset']=='handcrafted']


# In[7]:


# Storing the encryption keys for further decryption
enc_keys = []

# download Apple reports to save_dir
def download_files(df: pd.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for url in df['pdf_url']:
        pdf_filename = os.path.basename(url)
        # Checking if the file is encrypted
        if('?' in pdf_filename):
            # Saving the password for decryption
            enc_keys.append(pdf_filename)
            # Removing question mark
            pdf_filename = pdf_filename.split('?')[0]
            
        response = requests.get(url)
        with open(os.path.join(save_dir, pdf_filename), 'wb') as file:
            file.write(response.content)
    print(f"Success.")


# In[8]:


download_files(df_sample, REPORTS_SAVE_PATH)


# ## Create simple vector database

# In[9]:


def get_password(f):
    for tmp in enc_keys:
        if(f == tmp.split()[0]):
            return tmp

def get_documents_from_path(files_path: str) -> [Document]:
    documents = []
    
    for file in os.listdir(files_path):
        _, file_extension = os.path.splitext(file)
        text = ""
        
        if file_extension == ".pdf":
            with open(os.path.join(files_path, file), 'rb') as f:
                reader = PyPDF2.PdfReader(f, strict=False)
                
                if reader.is_encrypted:
                    try:
                        # Try to decrypt with the provided password (or an empty string if no password is given)
                        pdf_password = get_password(file)
                        
                        if pdf_password:
                            success = reader.decrypt(pdf_password)
                        else:
                            success = reader.decrypt("")

                        if success == 0:
                            print(f"Failed to decrypt {file}: Invalid password.")
                            continue  # Skip file if decryption fails
                        else:
                            print(f"Decrypted {file} successfully.")
                    except Exception as e:
                        print(f"Failed to decrypt {file}: {e}")
                        continue  # Skip file if decryption fails
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
            if text:
                documents.append(Document(page_content=text, metadata={"source": file}))
            else:
                print(f"WARNING: No text extracted from {file}")
        else:
            # TODO: can add support for other file types here
            raise Exception(f"Unsupported file extension: {file_extension}")
    
    return documents


# In[10]:


documents = get_documents_from_path(REPORTS_SAVE_PATH)


# In[11]:


# TODO could also just provide a dummy retriever to not spoil too much
class DummyRetriever:
    def __init__(self, texts):
        self.texts = texts
        
    def dummy_retriever(self, query):
        import random
        return random.sample(self.texts, k=3)


# In[12]:


# Create database
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300, separators=["\n\n", "\n"])

# split documents and create vector database
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()  # https://platform.openai.com/docs/guides/embeddings/embedding-models
db = FAISS.from_documents(texts, embeddings)

# count build embedding token number
tokenizer = tiktoken.get_encoding("cl100k_base")
build_token_count = sum([len(tokenizer.encode(doc.page_content)) for doc in texts])
print(f"Token count: {build_token_count}")


# In[14]:


# Store the database
with open(DB_PATH, "wb") as f:
    pickle.dump(db.serialize_to_bytes(), f)


# ## Create simple RAG

# In[15]:


# Load the database
DB_PATH = "data/db/sample.db"

with open(DB_PATH, "rb") as f:
    db_bytes = pickle.load(f)
    db = FAISS.deserialize_from_bytes(db_bytes, OpenAIEmbeddings(), allow_dangerous_deserialization=True)


# In[42]:


client = OpenAI()

retriever=db.as_retriever()

class Answer(BaseModel):
    value: Optional[List[Union[float, int]]]
    unit: str
    chain_of_thought: str

def retrieve_context(question):
    context_docs = retriever.get_relevant_documents(question)
    context = '\n'.join([doc.page_content for doc in context_docs])
    return context

def construct_messages(context, question):
    system_prompt = (
        "You are an expert assistant. Use only the following retrieved context to answer the question accurately and concisely. "
        "Provide your answer as a number followed by its unit, without any additional text or explanation. "
        "Before giving the final answer, include your chain-of-thought reasoning prefixed with 'Chain of Thought:'. "
        "If nothing is mentioned in the context, say 'I don't know'."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    return messages

def get_response_from_openai(question,verbose=True):
    context = retrieve_context(question)
    messages = construct_messages(context, question)
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=messages,
        response_format=Answer
        )
    answer = completion.choices[0].message.parsed

    if verbose:
        if answer.value is not None:
            print(f"Value: {answer.value}")
            print(f"Unit: {answer.unit}")
            print(f"Chain of Thought: {answer.chain_of_thought}")
        else:
            print("Cannot answer.")
            print(f"Chain of Thought: {answer.chain_of_thought}")

    return answer


# In[28]:




print("Companies under the scope of our reports:")
    
# Assuming there is a 'Company' column in df
companies = df_sample['company_name'].unique()  # Get unique company names
for company in companies:
    print(f"- {company}")

# Step 2: Ask the user for questions in a loop
while True:
    question = input("\nPlease enter a question (or type 'stop' to exit): ")

    # Step 3: Check if the user wants to stop
    if question.lower() == "stop":
        print("Exiting the question loop. Goodbye!")
        break

    # Step 4: Use the `get_response_from_openai()` function to get and display the answer
    try:
        answer = get_response_from_openai(question, verbose=True)
    except Exception as e:
        print(f"An error occurred: {e}")

