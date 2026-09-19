import os
import threading
import tkinter as tk
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
from langchain.memory import ChatMessageHistory


# === Setup RAG pipeline ===
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#r= raw string: tells python not to treat \ as escape sequences
file_name = r"fccu_catalogue_data (cleaned).txt"

#Text Loader loads contents of text files as LangChain Document/list of Langchain documents
loader = TextLoader(file_name, encoding = "utf-8")
data = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
data_chunks = splitter.split_documents(data)

index_path_name = "faiss_index"

if os.path.exists(index_path_name):
    vector_store = FAISS.load_local(index_path_name, embeddings=model, allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_documents(data_chunks, model)
    vector_store.save_local(index_path_name)

generator = OllamaLLM(model="mistral")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant designed to
answer the admission queries of potential/new
FCCU Students. Only answer on the basis of the given context and previous chat history. If you
cannot find the answer to a question on the basis of the given
context, respond with: "I don't know the answer to this question
because I don't have any information available related to this." But if the user
     says meow, then respond with meow."""),

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

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

speech_recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        speech_recognizer.adjust_for_ambient_noise(source)
        speech_audio = speech_recognizer.listen(source)
    try:
        return speech_recognizer.recognize_google(speech_audio)
    except:
        return "Sorry, I could not understand that."


# === GUI class ===
class ChatGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🤖 FCCU Admission Assistant")
        self.geometry("550x650")
        self.config(bg="#152F4A")

        self.chat_frame = tk.Frame(self, bg="#152F4A")
        self.chat_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.chat_canvas = tk.Canvas(self.chat_frame, bg="#152F4A", highlightthickness=0)
        self.chat_scrollbar = tk.Scrollbar(self.chat_frame, orient="vertical", command=self.chat_canvas.yview)
        self.chat_inner = tk.Frame(self.chat_canvas, bg="#152F4A")

        self.chat_inner.bind("<Configure>", lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all")))

        self.chat_canvas.create_window((0, 0), window=self.chat_inner, anchor="nw")
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)

        self.chat_canvas.pack(side="left", fill="both", expand=True)
        self.chat_scrollbar.pack(side="right", fill="y")

        self.input_frame = tk.Frame(self, bg="#152F4A")
        self.input_frame.pack(fill="x", padx=15, pady=10)

        self.user_input = tk.Entry(self.input_frame, font=("Segoe UI", 12), bg="#1E3A66", fg="white",
                                   insertbackground="white", relief="flat")
        self.user_input.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))
        self.user_input.bind("<Return>", lambda e: self.handle_input())

        self.mic_button = tk.Button(self.input_frame, text="🎤", font=("Segoe UI Emoji", 14), bg="#1E90FF",
                                    fg="white", relief="flat", command=self.handle_voice)
        self.mic_button.pack(side="left")

        self.send_button = tk.Button(self.input_frame, text="➤", font=("Segoe UI Emoji", 14), bg="#1E90FF",
                                     fg="white", relief="flat", command=self.handle_input)
        self.send_button.pack(side="left", padx=(10, 0))

        self.stop_button = tk.Button(self, text="Stop", bg="#D9534F", fg="white",
                                     font=("Segoe UI", 12, "bold"), command=self.stop_response)
        self.stop_button.pack(pady=(0, 10))
        self.stop_button.pack_forget()

        self.is_generating = False
        self.stop_flag = False

    def update_chat(self, message, sender="user"):
        bubble_color = "#ADD8E6" if sender == "user" else "#4A90E2"
        anchor = "e" if sender == "user" else "w"
        justify = "right" if sender == "user" else "left"

        bubble = tk.Label(self.chat_inner, text=message, bg=bubble_color, fg="black",
                          font=("Segoe UI", 11), wraplength=450, justify=justify, padx=12, pady=8)
        bubble.pack(anchor=anchor, fill="none", pady=4, padx=10)
        self.chat_canvas.yview_moveto(1)

    def handle_input(self):
        if self.is_generating:
            return

        user_text = self.user_input.get().strip()
        if not user_text:
            return

        self.update_chat(user_text, "user")
        self.user_input.delete(0, "end")

        threading.Thread(target=self.get_response, args=(user_text,), daemon=True).start()

    def handle_voice(self):
        if self.is_generating:
            return
        self.update_chat("Listening...", "system")
        threading.Thread(target=self.voice_to_response, daemon=True).start()

    def voice_to_response(self):
        question = listen()
        self.update_chat(question, "user")
        self.get_response(question)

    def get_response(self, question):
        self.is_generating = True
        self.stop_flag = False
        self.stop_button.pack(pady=(0, 10))

        try:
            retrieved_docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)

            if not context.strip():
                response = "🤖 I’m sorry, I don’t know the answer to that based on the available information."
            else:
                result = chain_with_history.invoke(
                    {
                        "input": question,
                        "context": context
                    },
                    config={"configurable": {"session_id": "test-session"}}
                )
                response = result.get("answer", "⚠️ Unexpected response format.")

            if self.stop_flag:
                response = "🤖 Response generation stopped."

            self.update_chat(response, "bot")

        except Exception as e:
            self.update_chat(f"❌ Error: {e}", "bot")

        self.is_generating = False
        self.stop_flag = False
        self.stop_button.pack_forget()

    def stop_response(self):
        if self.is_generating:
            self.stop_flag = True
            self.update_chat("Response generation stopped by user.", "system")
            self.is_generating = False
            self.stop_button.pack_forget()


if __name__ == "__main__":
    app = ChatGUI()
    app.mainloop()
