import os
import threading
import speech_recognition as sr

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# === Setup RAG pipeline ===
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

files = [
    "fccu_catalogue_data (cleaned).txt",
    "fccu_bs_admissions (cleaned).txt",
    "fccu_residential_life (cleaned).txt",
    "fccu_tuition_fee (cleaned).txt",
    "fccu_scholarships (cleaned).txt",
    "fccu_postgraduate_programs (cleaned).txt",
    "fccu_pharmad_admissions (cleaned).txt",
    "fccu_hostel_policy (cleaned).txt"
]

data = []
base_path = r"D:\FCCU Admission Assistant (Final)"
for file in files:
    path = os.path.join(base_path, file)
    loader = TextLoader(path, encoding="utf-8")
    data += loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=250, separators=["\n\n", "\n", ".", " "])
data_chunks = splitter.split_documents(data)

index_path_name = "faiss_index"
if os.path.exists(index_path_name):
    vector_store = FAISS.load_local(index_path_name, embeddings=model, allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_documents(data_chunks, model)
    vector_store.save_local(index_path_name)

generator = OllamaLLM(model="mistral")
retriever = vector_store.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are an assistant designed to answer the admission queries of potential/new FCCU (Forman Christian College University) Students.
        Use the conversation history (chat_history) to understand follow-up questions.
        Only answer on the basis of the given context and previous chat history.
        If you cannot find the answer to a question on the basis of the given context, respond with:
        "I don't know the answer to this question because I don't have any information available related to this."
        If asked about postgraduate majors/master programs, list the programs mentioned in context.
        If the user asks something like "explain this in more detail", assume they are referring to your last answer.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context:\n{context}\n\nQuestion: {input}")
])

documents_chain = create_stuff_documents_chain(llm=generator, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, documents_chain)

chats_store = {}

def get_session_history(session_id):
    if session_id not in chats_store:
        chats_store[session_id] = ChatMessageHistory()
    return chats_store[session_id]

history_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

speech_recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        print("🎙️ Listening...")
        speech_recognizer.adjust_for_ambient_noise(source)
        speech_audio = speech_recognizer.listen(source)
    try:
        return speech_recognizer.recognize_google(speech_audio)
    except:
        return "Sorry, I could not understand that."

def retrieve_documents(question, session_id, turns=2):
    chat_history_obj = get_session_history(session_id)
    is_first_question = len(chat_history_obj.messages) == 0

    history_context = "\n".join([
        f"User: {msg.content}" if msg.type == "human" else f"Assistant: {msg.content}"
        for msg in chat_history_obj.messages[-turns:]
    ])

    retrieved_docs = []
    if is_first_question:
        retrieved_docs = retriever.invoke(question, search_kwargs={"k": 4})
    else:
        if history_context.strip():
            docs_from_history = retriever.invoke(history_context, search_kwargs={"k": 2})
            retrieved_docs.extend(docs_from_history)

        docs_from_query = retriever.invoke(question, search_kwargs={"k": 2})
        retrieved_docs.extend(docs_from_query)

    return retrieved_docs

def chat_loop():
    session_id = "test-session"
    last_bot_response = ""

    print("\n📘 FCCU Admission Assistant (Text/Voice Mode)\nType 'voice' to speak | Type 'exit' to quit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if user_input.lower() == "voice":
            user_input = listen()
            print(f"You (via Voice): {user_input}")

        vague_phrases = ["explain this", "explain this in more detail", "what do you mean", "clarify this"]
        if user_input.lower() in vague_phrases and last_bot_response:
            user_input = f"Can you explain this more clearly: {last_bot_response}"

        chat_history_obj = get_session_history(session_id)
        chat_history_obj.add_user_message(user_input)

        retrieved_docs = retrieve_documents(user_input, session_id)
        retrieved_texts = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

        print("\n📄 Retrieved Context:\n")
        print(retrieved_texts)
        print("\n============================\n")

        try:
            result = history_chain.invoke(
                {"input": user_input, "context": retrieved_texts},
                config={"configurable": {"session_id": session_id}}
            )
            response = result.get("answer", "⚠️ Unexpected response format.")
            last_bot_response = response
            chat_history_obj.add_ai_message(response)

            print(f"🤖 Assistant: {response}\n")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    chat_loop()





