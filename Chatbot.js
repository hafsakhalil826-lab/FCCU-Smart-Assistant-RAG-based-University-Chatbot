
//Code to fetch API

const sendBtn = document.getElementById("send-btn");
const messageInput = document.getElementById("message-input");
const chatWindow = document.getElementById("chat-window");
const clearBtn = document.getElementById("clear-btn");
const micBtn = document.getElementById("mic-btn");

// Optional: Generate a unique session_id for each session
const sessionId = `user-session-${Date.now()}`;

// Helper function to add user message bubble
function addUserMessage(text) {
  const userMessage = document.createElement('div');
  userMessage.className = 'message user-message';

  const userBox = document.createElement('div');
  userBox.className = 'user-box';

  const paragraph = document.createElement('p');
  paragraph.textContent = text;

  userBox.appendChild(paragraph);
  userMessage.appendChild(userBox);

  chatWindow.appendChild(userMessage);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Helper function to add bot message bubble
function addBotMessage(text) {
  const botMessage = document.createElement('div');
  botMessage.className = 'message bot-message';

  const insideBox = document.createElement('div');
  insideBox.className = 'inside-box';

  const paragraph = document.createElement('p');
  paragraph.textContent = text;

  insideBox.appendChild(paragraph);
  botMessage.appendChild(insideBox);

  chatWindow.appendChild(botMessage);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}



function fetchBotResponse(userInput) {
  addBotMessage("🤖 Thinking...");

  fetch('http://127.0.0.1:5000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_question: userInput,    
      session_id: sessionId         
    })
  })
  .then(res => res.json())
  .then(data => {
    chatWindow.lastChild.remove();  
    addBotMessage(data.response);  // display bot response
  })
  .catch(err => {
    chatWindow.lastChild.remove();
    addBotMessage("Error: Could not reach server.");
    console.error(err);
  });
}


// Send button click handler
sendBtn.addEventListener('click', () => {
  const messageText = messageInput.value.trim();
  if (messageText !== '') {
    addUserMessage(messageText);
    messageInput.value = '';
    fetchBotResponse(messageText);
  }
});

// Clear button removes all messages except original greeting
clearBtn.addEventListener('click', () => {
  const allMessages = chatWindow.querySelectorAll(".message");
  allMessages.forEach((msg, index) => {
    if (index !== 0) msg.remove(); // Keep first message
  });
});

// SPEECH RECOGNITION
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
  const recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  micBtn.addEventListener('click', () => {
    recognition.start();
  });

  recognition.addEventListener('result', (event) => {
    const speechResult = event.results[0][0].transcript;
    if (speechResult.trim()) {
      addUserMessage(speechResult);
      fetchBotResponse(speechResult);
    }
  });

  recognition.addEventListener('speechend', () => {
    recognition.stop();
  });

  recognition.addEventListener('error', (event) => {
    console.error('Speech recognition error:', event.error);
  });

} else {
  micBtn.disabled = true;
  alert("Speech Recognition is not supported in your browser.");
}
