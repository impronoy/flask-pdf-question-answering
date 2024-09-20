from flask import Flask, render_template, request, redirect, url_for
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

UPLOAD_FOLDER = 'uploads/'
INDEX_FOLDER = 'faiss_index/'

# Create directories if they don't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(INDEX_FOLDER):
    os.makedirs(INDEX_FOLDER)

# Configure the Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(INDEX_FOLDER)  # Save FAISS index to specified folder
    return vector_store

# Load a conversational chain
def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant tasked with answering questions based on the given context. 
    Please provide thorough and accurate answers based on the context provided. 
    If the answer is not within the context, state "The answer is not available in the provided context."
    Do not attempt to answer if the information is not present.

    Context: {context}

    Question: {question}

    Answer in detail:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Retrieve documents from FAISS index
def retrieve_documents(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    return docs

# Generate answers from retrieved documents
def generate_answer(docs, user_question):
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Home route to render the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle PDF upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Extract text from the uploaded PDF
        pdf_text = get_pdf_text([file])
        text_chunks = get_text_chunks(pdf_text)
        
        # Generate vector store from text chunks
        vector_store = get_vector_store(text_chunks)

        # Save the uploaded file
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))

        # Redirect to ask a question
        return redirect(url_for('ask_question'))

# Route to ask questions after uploading PDF
@app.route('/ask', methods=['GET', 'POST'])
def ask_question():
    if request.method == 'POST':
        question = request.form['question']

        # Load the vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)

        # Retrieve documents based on the question
        docs = retrieve_documents(question, vector_store)

        # Generate the answer
        answer = generate_answer(docs, question)
        
        return render_template('index.html', answer=answer)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
