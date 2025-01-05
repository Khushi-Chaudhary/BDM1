import { createStore } from "vuex";

export default createStore({
  state: {
    chatHistory: [],
    emailValidated: false,
    email: "",
    name: "",
  },
  mutations: {
    addChat(state, chat) {
      state.chatHistory.push(chat);
    },
    validateEmail(state, email) {
      state.emailValidated = true;
      state.email = email;
    },
    setName(state, name) {
      state.name = name;
    },
    resetSession(state) {
      state.chatHistory = [];
      state.emailValidated = false;
      state.email = "";
      state.name = "";
    },
  },
});
