<template>
  <div id="app">
    <BDMChatbot />
  </div>
</template>

<script>
import BDMChatbot from './components/BDMChatbot.vue';

export default {
  name: 'App',
  components: {
    BDMChatbot,
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 20px;
}
</style>
