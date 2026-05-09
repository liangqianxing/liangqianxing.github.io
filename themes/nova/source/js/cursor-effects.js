(function () {
  'use strict';

  const finePointer = window.matchMedia && window.matchMedia('(pointer: fine)').matches;
  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!finePointer || reduceMotion) return;

  const root = document.documentElement;
  const dot = document.createElement('div');
  const ring = document.createElement('div');
  const glow = document.createElement('div');
  dot.className = 'nova-cursor-dot';
  ring.className = 'nova-cursor-ring';
  glow.className = 'nova-cursor-glow';
  document.body.append(ring, glow, dot);
  root.classList.add('nova-custom-cursor');

  let mouseX = window.innerWidth / 2;
  let mouseY = window.innerHeight / 2;
  let ringX = mouseX;
  let ringY = mouseY;
  let glowX = mouseX;
  let glowY = mouseY;
  let raf = 0;
  let idleTimer = 0;

  const interactiveSelector = 'a, button, input, textarea, select, summary, label, [role="button"], .post-card, .project-card, .social-link, .nav-link, .theme-toggle, .ac-post-item, .ac-edu-item';

  function setVisible(visible) {
    root.classList.toggle('nova-cursor-hidden', !visible);
  }

  function setIdle() {
    root.classList.add('nova-cursor-idle');
  }

  function wake() {
    root.classList.remove('nova-cursor-idle');
    window.clearTimeout(idleTimer);
    idleTimer = window.setTimeout(setIdle, 1800);
  }

  function move(event) {
    mouseX = event.clientX;
    mouseY = event.clientY;
    dot.style.transform = `translate3d(${mouseX}px, ${mouseY}px, 0) translate(-50%, -50%)`;
    setVisible(true);
    wake();
  }

  function animate() {
    ringX += (mouseX - ringX) * 0.22;
    ringY += (mouseY - ringY) * 0.22;
    glowX += (mouseX - glowX) * 0.09;
    glowY += (mouseY - glowY) * 0.09;
    ring.style.transform = `translate3d(${ringX}px, ${ringY}px, 0) translate(-50%, -50%)`;
    glow.style.transform = `translate3d(${glowX}px, ${glowY}px, 0) translate(-50%, -50%)`;
    raf = window.requestAnimationFrame(animate);
  }

  document.addEventListener('pointermove', move, { passive: true });
  document.addEventListener('pointerleave', () => setVisible(false), { passive: true });
  document.addEventListener('pointerenter', () => setVisible(true), { passive: true });
  document.addEventListener('pointerdown', () => root.classList.add('nova-cursor-pressed'), { passive: true });
  document.addEventListener('pointerup', () => root.classList.remove('nova-cursor-pressed'), { passive: true });

  document.addEventListener('mouseover', (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    root.classList.toggle('nova-cursor-hover', Boolean(target.closest(interactiveSelector)));
    root.classList.toggle('nova-cursor-text', Boolean(target.closest('p, li, h1, h2, h3, h4, h5, h6, blockquote, code, pre')) && !target.closest(interactiveSelector));
  }, { passive: true });

  window.addEventListener('blur', () => setVisible(false));
  window.addEventListener('focus', () => setVisible(true));
  window.addEventListener('pagehide', () => {
    window.cancelAnimationFrame(raf);
    window.clearTimeout(idleTimer);
  }, { once: true });

  dot.style.transform = `translate3d(${mouseX}px, ${mouseY}px, 0) translate(-50%, -50%)`;
  animate();
  wake();
})();
