<template>
  <div id="app">
    <app-header></app-header>
    <app-content></app-content>
    <app-footer></app-footer>
  </div>
</template>

<script>
  import axios from "axios";
  import Header from "./components/Header";
  import Footer from "./components/Footer";
  import Content from "./components/Content";
  export default {
    name: "App",
    data() {
      return {};
    },
    components:{
      'app-header':Header,
      'app-content':Content,
      'app-footer':Footer},
    methods: {} ,

  };
</script>
