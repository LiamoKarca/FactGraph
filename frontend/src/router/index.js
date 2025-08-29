import { createRouter, createWebHistory } from 'vue-router'

import ServicePage  from '../components/ServicePage.vue'
import TaskStatus   from '../views/TaskStatus.vue'
import ServiceIntro from '../pages/ServiceIntro.vue'
import AboutUs      from '../pages/AboutUs.vue'

const APP_NAME = '芒狗偵探'
const SEP = '｜'

const routes = [
  { path: '/',          name: 'home',    component: ServicePage,  meta: { title: '' } },
  { path: '/service',   name: 'service', component: ServiceIntro, meta: { title: '服務介紹' } },
  { path: '/about',     name: 'about',   component: AboutUs,      meta: { title: '關於我們' } },
  { path: '/tasks/:id', name: 'task',    component: TaskStatus,   meta: { title: '查核結果' } },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// 統一設定：芒狗偵探｜<頁名>（APP 在左，頁名在右）
router.afterEach((to) => {
  const raw = typeof to.meta?.title === 'function' ? to.meta.title(to) : to.meta?.title
  document.title = raw ? `${APP_NAME}${SEP}${raw}` : APP_NAME
})

export default router
export { router }
