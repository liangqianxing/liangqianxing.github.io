/* Nova Theme JS */
(function () {
  'use strict';

  // Theme toggle
  const html = document.documentElement;
  const toggleBtn = document.getElementById('themeToggle');
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const saved = localStorage.getItem('nova-theme');
  const initialTheme = saved || (prefersDark ? 'dark' : 'light');
  html.setAttribute('data-theme', initialTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const current = html.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);
      localStorage.setItem('nova-theme', next);
    });
  }

  // Mobile nav
  const hamburger = document.getElementById('navHamburger');
  const mobileNav = document.getElementById('navMobile');
  if (hamburger && mobileNav) {
    hamburger.addEventListener('click', () => {
      mobileNav.classList.toggle('open');
    });
  }

  // Nav scroll shadow
  const nav = document.getElementById('nav');
  if (nav) {
    window.addEventListener('scroll', () => {
      nav.style.boxShadow = window.scrollY > 10 ? '0 2px 12px rgba(0,0,0,.08)' : '';
    }, { passive: true });
  }

  function initSearch() {
    const modal = document.getElementById('searchModal');
    const input = document.getElementById('searchInput');
    const results = document.getElementById('searchResults');
    const meta = document.getElementById('searchMeta');
    const openButtons = [document.getElementById('searchToggle'), document.getElementById('mobileSearchToggle')].filter(Boolean);
    if (!modal || !input || !results || !meta || !openButtons.length) return;

    let posts = [];
    let loadingPromise = null;

    const stripHtml = value => String(value || '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
    const escapeHtml = value => String(value || '').replace(/[&<>'"]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
    const normalize = value => stripHtml(value).toLowerCase();

    function loadIndex() {
      if (loadingPromise) return loadingPromise;
      meta.textContent = '正在加载搜索索引...';
      loadingPromise = fetch('/search.json')
        .then(response => {
          if (!response.ok) throw new Error('Search index not found');
          return response.json();
        })
        .then(data => {
          posts = (Array.isArray(data) ? data : []).map(post => ({
            title: stripHtml(post.title),
            url: post.url || post.path || '#',
            content: stripHtml(post.content),
            searchText: normalize(`${post.title || ''} ${post.content || ''}`)
          }));
          meta.textContent = posts.length ? '输入关键词开始搜索' : '暂无可搜索文章';
        })
        .catch(() => {
          meta.textContent = '搜索索引加载失败，请先重新生成站点';
        });
      return loadingPromise;
    }

    function render(query) {
      const keywords = normalize(query).split(/\s+/).filter(Boolean);
      results.innerHTML = '';
      if (!keywords.length) {
        meta.textContent = posts.length ? '输入关键词开始搜索' : '暂无可搜索文章';
        return;
      }

      const matches = posts
        .map(post => {
          const score = keywords.reduce((sum, keyword) => sum + (post.searchText.includes(keyword) ? 1 : 0), 0);
          return { post, score };
        })
        .filter(item => item.score === keywords.length)
        .slice(0, 20);

      meta.textContent = matches.length ? `找到 ${matches.length} 条结果` : '没有找到相关文章';
      results.innerHTML = matches.map(({ post }) => {
        const firstKeyword = keywords[0];
        const index = post.content.toLowerCase().indexOf(firstKeyword);
        const excerptStart = Math.max(0, index - 45);
        const excerpt = post.content.slice(excerptStart, excerptStart + 140) || post.title;
        return `<a class="search-result-item" href="${escapeHtml(post.url)}"><span class="search-result-title">${escapeHtml(post.title || '无标题')}</span><span class="search-result-excerpt">${escapeHtml(excerpt)}${post.content.length > excerptStart + 140 ? '...' : ''}</span></a>`;
      }).join('');
    }

    function openSearch() {
      modal.classList.add('open');
      modal.setAttribute('aria-hidden', 'false');
      document.body.classList.add('search-open');
      loadIndex().then(() => render(input.value));
      setTimeout(() => input.focus(), 50);
      if (mobileNav) mobileNav.classList.remove('open');
    }

    function closeSearch() {
      modal.classList.remove('open');
      modal.setAttribute('aria-hidden', 'true');
      document.body.classList.remove('search-open');
    }

    openButtons.forEach(button => button.addEventListener('click', openSearch));
    modal.querySelectorAll('[data-search-close]').forEach(button => button.addEventListener('click', closeSearch));
    input.addEventListener('input', () => render(input.value));
    document.addEventListener('keydown', event => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        openSearch();
      } else if (event.key === 'Escape' && modal.classList.contains('open')) {
        closeSearch();
      }
    });
  }

  initSearch();

  // Build TOC dynamically from actual heading IDs in the page
  function buildToc() {
    const tocList = document.getElementById('tocList');
    if (!tocList) return;
    const headings = Array.from(document.querySelectorAll('.post-content h1, .post-content h2, .post-content h3'));
    if (!headings.length) return;

    headings.forEach(h => {
      // id may be on the heading itself or on an inner <span>
      const idEl = h.id ? h : h.querySelector('[id]');
      if (!idEl) return;
      const id = idEl.id;
      const level = parseInt(h.tagName[1]);
      const li = document.createElement('li');
      li.className = `toc-item toc-level-${level}`;
      const a = document.createElement('a');
      a.className = 'toc-link';
      a.href = '#' + id;
      a.textContent = h.textContent.trim();
      li.appendChild(a);
      tocList.appendChild(li);
    });

    // Highlight active on scroll
    const tocLinks = tocList.querySelectorAll('a');
    const onScroll = () => {
      const scrollY = window.scrollY + 100;
      let active = headings[0];
      for (const h of headings) {
        if (h.offsetTop <= scrollY) active = h;
      }
      tocLinks.forEach(a => a.classList.remove('active'));
      if (active) {
        const idEl = active.id ? active : active.querySelector('[id]');
        if (idEl) {
          const match = tocList.querySelector(`a[href="#${CSS.escape(idEl.id)}"]`);
          if (match) match.classList.add('active');
        }
      }
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', buildToc);
  } else {
    buildToc();
  }
})();
