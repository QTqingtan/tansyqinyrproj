import Vue from 'vue'
import Router from 'vue-router'

import Layout from "../components/Layout.vue";
import Upload2Show from "@/components/Upload2Show.vue";
import prepic from "@/components/prepic.vue";
Vue.use(Router)

const routess = [];

// 对应我们的50个操作
for (let id = 0; id < 40; id++) {
      const route = {
          path: `/upload/:id`,
          // 如果用${id}是使用模板字符
          // 如果用:id是使用参数定义!!
          components: {
              default: Upload2Show,
              footer: prepic,
          },
          props: true,
          meta: { id: id}, // 可以通过this.$route.meta.id获取这个值
          //新增监听路由 因为使用同一个component
          // 在Upload2Show组件中增加了watch
          // 切换页面需要 组件要刷新!
      };
      routess.push(route);
}

const routesss = [];

// 对应我们的50个操作
for (let id = 51; id <= 59; id++) {
      const route = {
          path: `/upload/:id`,
          components: {
              default: Upload2Show,
              footer: prepic,
          },
          props: true,
          meta: { id: id},
      };
      routesss.push(route);
}

const routes = [
    {
        path: '/',
        redirect:'/upload/50',
        component: Layout,
        hidden: true,
    },
    {
        path: '/upload',
        component: Layout,
        hidden: true,
        children: [
            {//TODO 还不知道为什么这个要放在上面
                path: '/upload/50',
                component: () => import("../views/50OCR.vue"),
                name: 'OCR',
            },
            ...routess,
            ...routesss,
        ]
    },
];

const router = new Router({
    mode: 'history',//取消哈希模式
    routes
});

export default router;
