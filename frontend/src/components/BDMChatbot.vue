<template>
  <div class="chatbot">
    <h1>BDM Chatbot</h1>
    <p>
      Developed by Lalit & Puneet, students of BS (Applications & Data Science)
      IIT Madras
    </p>
    <p>
      We encourage you to ask clear and relevant questions about BDM project
      only to ensure meaningful responses.
    </p>
    <p>
      <strong>Disclaimer:</strong> All interactions are recorded. By using this
      chatbot, you agree to the storage of this data.
    </p>

    <!-- Email Input -->
    <div v-if="!emailValidated">
      <input
        v-model="email"
        type="text"
        placeholder="Enter your email (format: XXfXXXXXXX@ds.study.iitm.ac.in)"
      />
      <button @click="validateEmail">Validate Email</button>
      <p v-if="emailError" class="error">{{ emailError }}</p>
    </div>

    <!-- Chat Section -->
    <div v-else>
      <input
        v-model="name"
        type="text"
        placeholder="Enter your name (optional)"
      />
      <textarea
        v-model="userInput"
        placeholder="Pose your Questions..."
        @keyup.enter="sendQuestion"
      ></textarea>
      <button @click="sendQuestion">Ask</button>
      <button @click="endChat">Stop</button>

      <!-- Chat History -->
      <div class="chat-history">
        <h3>Chat History</h3>
        <div v-for="(chat, index) in chatHistory" :key="index" class="chat">
          <p>
            <strong>Q{{ index + 1 }}:</strong> {{ chat.question }}
          </p>
          <p><strong>Chatbot:</strong> {{ chat.answer }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from "axios";
export default {
  data() {
    return {
      email: "",
      name: "",
      emailValidated: false,
      userInput: "",
      chatHistory: [],
      emailError: "",
    };
  },
  methods: {
    validateEmail() {
      const emailRegex = /^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$/;
      if (emailRegex.test(this.email)) {
        this.emailValidated = true;
        this.emailError = "";
      } else {
        this.emailError = "Invalid email format. Please enter a valid email.";
      }
    },
    async sendQuestion() {
      if (!this.userInput.trim()) return;

      const question = this.userInput.trim();
      this.chatHistory.push({ question, answer: "Loading..." });

      try {
        const response = await axios.post("http://localhost:5000/ask", {
          question,
          chatHistory: this.chatHistory,
        });
        const answer = response.data.answer;
        this.chatHistory[this.chatHistory.length - 1].answer = answer;
      } catch (error) {
        this.chatHistory[this.chatHistory.length - 1].answer =
          "Error fetching response. Please try again.";
      }

      this.userInput = "";
    },
    endChat() {
      alert("Chat session ended. Thank you!");
      this.emailValidated = false;
      this.chatHistory = [];
      this.email = "";
      this.name = "";
    },
  },
};
</script>

<style scoped>
.chatbot {
  max-width: 600px;
  margin: auto;
  font-family: Arial, sans-serif;
}
textarea {
  width: 100%;
  height: 100px;
  margin: 10px 0;
}
button {
  margin: 5px;
}
.chat-history {
  margin-top: 20px;
  background: #f9f9f9;
  padding: 10px;
  border: 1px solid #ddd;
}
.error {
  color: red;
}
</style>
