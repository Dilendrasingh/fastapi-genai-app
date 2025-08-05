import os
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
app = FastAPI()
load_dotenv() 
UPLOAD_DIR = "uploaded_pdfs"
VECTORSTORE_DIR = "vectorstore/db_faiss"
os.makedirs(UPLOAD_DIR, exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM Setup
llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    google_api_key=GOOGLE_API_KEY
)

CUSTOM_PROMPT_TEMPLATE = """
You are an  expert in reading pdfs and extract infromation from it.

Use only the information provided in the context to answer the question. If the answer is not found in the context, respond with: "The information is not available in the provided context."

Use **paragraph format** for definitions or general explanations.  
Use **bullet points** only when listing steps, advantages, roles, features, or other multi-point responses.

Do not speculate.

If the user explicitly requests the answer in Nepali (e.g., says "explain in Nepali", "in Nepali", "answer in Nepali" or similar), then:
- If the context is in English, translate your answer into Nepali before responding.
- If the context is already in Nepali, answer in Nepali directly.

Otherwise, respond in English by default.

Context:
{context}

User Question:
{question}

Answer:
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

# PDF Upload Endpoint
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"[INFO] PDF saved: {file_path}")

        # Load & split PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"[INFO] Loaded {len(documents)} documents from PDF.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=70)
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split into {len(chunks)} chunks.")

        # Generate and save FAISS vectorstore
        embedding_model = get_embedding_model()
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(VECTORSTORE_DIR)
        print(f"[INFO] Vectorstore saved at {VECTORSTORE_DIR}")

        return {"message": f"{file.filename} uploaded and processed successfully."}

    except Exception as e:
        tb = traceback.format_exc()
        print("[ERROR] Upload failed:", tb)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

# Memory for chat
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Q&A Endpoint
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    try:
        if not os.path.exists(VECTORSTORE_DIR):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No vectorstore found. Upload a PDF first."
            )

        embedding_model = get_embedding_model()
        db = FAISS.load_local(VECTORSTORE_DIR, embedding_model, allow_dangerous_deserialization=True)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 7}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        result = qa_chain.invoke({"question": question})
        return {"answer": result["answer"]}

    except Exception as e:
        tb = traceback.format_exc()
        print("[ERROR] Ask failed:", tb)
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")
