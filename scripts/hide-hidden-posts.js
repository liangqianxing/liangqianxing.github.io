'use strict';

hexo.extend.filter.register('before_generate', function () {
  const locals = this.locals;
  const posts = locals.get('posts');
  if (!posts || typeof posts.filter !== 'function') return;

  const visiblePosts = posts.filter(post => !post.hidden);
  locals.set('posts', visiblePosts);
});
