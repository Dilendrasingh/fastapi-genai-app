import os
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# === Load .env ===
load_dotenv()

# === FastAPI app ===
app = FastAPI()

# === CORS Configuration ===
origins = [
    "http://localhost:5173",
    "https://soeintelligence.vercel.app",
    "https://soeintel.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Constants & Setup ===
UPLOAD_DIR = "uploaded_pdfs"
VECTORSTORE_DIR = "vectorstore/db_faiss"
os.makedirs(UPLOAD_DIR, exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# === LLM and Embeddings ===
llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    google_api_key=GOOGLE_API_KEY
)

def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === Prompt Templates ===
CUSTOM_PROMPT_TEMPLATE = """
You are an expert in reading PDFs and extracting information from them.
If question like "who are you" or "who invented you", just answer: "I am invented by SOEIntel."
Replace any mention of Gemini or Google with SOEIntel in your responses.

Use only the information provided in the context to answer the question.
If the answer is not found in the context, respond: "The information is not available in the provided context."

Use **paragraph format** for definitions or general explanations.
Use **bullet points** for lists like steps, features, or advantages.

If asked in Nepali, respond in Nepali. Otherwise, respond in English.

Context:
{context}

User Question:
{question}

Answer:
"""

MCQ_PROMPT_TEMPLATE = """
You are an AI tutor helping students revise from academic content.

Your task is to generate {count} multiple-choice questions from the given context. For each question:
- Provide 4 options
- Clearly mention the correct answer
- Keep language academic and clear

Context:
{context}

Output format:
Q1. <question>
A. Option 1
B. Option 2
C. Option 3
D. Option 4
Answer: <correct_option_letter>
"""

SUMMARIZE_PROMPT_TEMPLATE = """
You are an expert summarizer. Read the context and generate a concise and informative summary in paragraph form.

Context:
{context}

Summary:
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def load_vectorstore():
    if not os.path.exists(VECTORSTORE_DIR):
        raise HTTPException(status_code=400, detail="No vectorstore found. Upload a PDF first.")
    embedding_model = get_embedding_model()
    return FAISS.load_local(VECTORSTORE_DIR, embedding_model, allow_dangerous_deserialization=True)

# === Endpoint: Upload PDF ===
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=70)
        chunks = splitter.split_documents(documents)

        embedding_model = get_embedding_model()
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(VECTORSTORE_DIR)

        return {"message": f"{file.filename} uploaded and processed successfully."}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

# === Endpoint: Ask Question ===
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    try:
        db = load_vectorstore()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 7}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )
        result = qa_chain.invoke({"question": question})
        return {"answer": result["answer"]}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

# === Endpoint: Generate MCQs ===
@app.post("/generate_mcq")
async def generate_mcq(count: int = Form(...)):
    try:
        db = load_vectorstore()
        context_docs = db.similarity_search("summary", k=10)
        context_text = "\n".join([doc.page_content for doc in context_docs])

        prompt = MCQ_PROMPT_TEMPLATE.format(context=context_text, count=count)
        response = llm.invoke(prompt)

        raw_text = response.content.strip()

        mcqs = []
        for block in raw_text.split("Q")[1:]:
            lines = block.strip().splitlines()
            if len(lines) < 6:
                continue
            question = lines[0][3:].strip()
            options = [line[3:].strip() for line in lines[1:5]]
            answer = lines[5].split(":")[-1].strip()
            mcqs.append({
                "question": question,
                "options": options,
                "answer": answer
            })

        return {"mcqs": mcqs}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate MCQs: {str(e)}")

# === Endpoint: Summarize PDF ===
@app.get("/summarize_pdf")
async def summarize_pdf():
    try:
        db = load_vectorstore()
        docs = db.similarity_search("summary", k=10)
        context_text = "\n".join([doc.page_content for doc in docs])
        summary_prompt = SUMMARIZE_PROMPT_TEMPLATE.format(context=context_text)
        response = llm.invoke(summary_prompt)
        return {"summary": response.content.strip()}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")
