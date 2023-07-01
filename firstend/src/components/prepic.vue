<template>
    <div class="demo-image__placeholder">
        <el-divider />
        <el-card>
             <div slot="header" class="clearfix">
            <span>效果呈现</span></div>
        <el-image :src="src0" style="width: 450px"></el-image>
        <el-image :src="src" style="width: 450px"></el-image>
        </el-card>
    </div>
</template>

<script>
import axios from "axios";

export default {
    name: "prepic",
    // props: ['id'],

    data() {
        return {
            src0: require('../../public/img/default.jpg'),
            src: require('../../public/img/'+this.$route.params.id+'.jpg')
        };
    },
    created: function () {
        // console.log('../../public/img/'+this.id+'.jpg');
        // console.log(this.id) undefined
        console.log(this.$route.params.id);
    },

    methods: {

  },
  mounted() {},
};
</script>

<style>

</style>

