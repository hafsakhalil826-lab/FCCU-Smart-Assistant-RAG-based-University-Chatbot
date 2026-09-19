# flask- request mainly used to extract data sent in http request

from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_chatbot import history_chain, retriever  # Your chatbot logic here
from langchain_community.chat_message_histories import ChatMessageHistory


app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_question = data.get("user_question")
    session_id = data.get("session_id")

    if not user_question:
        return jsonify({"response": " No question provided"}), 400

    try:
        retrieved_documents = retriever.invoke(user_question)
        context = "\n\n\n".join(doc.page_content for doc in retrieved_documents)

        response = history_chain.invoke(
            {
                "input": user_question,
                "context": context
            },
            config={"configurable": {"session_id": session_id}}
        )

        return jsonify({"response": response.get("answer")})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
