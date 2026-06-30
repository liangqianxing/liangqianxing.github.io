import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://liangqianxing.github.io',
  // 开启链接预取，鼠标悬停时提前加载目标页面，大幅提升页面切换速度
  prefetch: {
    prefetchAll: true,
    defaultStrategy: 'hover',
  },
  integrations: [tailwind(), sitemap()],
  markdown: {
    shikiConfig: {
      theme: 'github-dark-dimmed',
      wrap: true,
    },
  },
  build: {
    assets: 'assets',
    // 压缩 HTML 输出，减少传输体积
    inlineStylesheets: 'auto',
  },
});
