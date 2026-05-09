(function () {
  'use strict';

  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const coarsePointer = window.matchMedia && window.matchMedia('(pointer: coarse)').matches;
  const lowPower = navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4;

  if (reduceMotion || coarsePointer || lowPower || !window.THREE) return;

  const canvas = document.createElement('canvas');
  canvas.id = 'novaThreeBackdrop';
  canvas.setAttribute('aria-hidden', 'true');
  document.body.prepend(canvas);

  const THREE = window.THREE;
  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(58, 1, 0.1, 1000);
  camera.position.z = 34;

  const mouse = new THREE.Vector2(0, 0);
  const targetMouse = new THREE.Vector2(0, 0);
  const clock = new THREE.Clock();
  let width = 1;
  let height = 1;
  let animationId = 0;

  const palette = {
    light: {
      primary: new THREE.Color('#3b82f6'),
      secondary: new THREE.Color('#06b6d4'),
      accent: new THREE.Color('#8b5cf6'),
      opacity: 0.34,
      lineOpacity: 0.11,
    },
    dark: {
      primary: new THREE.Color('#8b5cf6'),
      secondary: new THREE.Color('#06b6d4'),
      accent: new THREE.Color('#22c55e'),
      opacity: 0.56,
      lineOpacity: 0.18,
    },
  };

  const getTheme = () => document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';

  const particleCount = 120;
  const positions = new Float32Array(particleCount * 3);
  const basePositions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);
  const speeds = new Float32Array(particleCount);

  function seedParticles() {
    const spreadX = 58;
    const spreadY = 36;
    for (let i = 0; i < particleCount; i += 1) {
      const i3 = i * 3;
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.sqrt(Math.random());
      const x = Math.cos(angle) * radius * spreadX;
      const y = Math.sin(angle) * radius * spreadY;
      const z = (Math.random() - 0.5) * 28;
      basePositions[i3] = x;
      basePositions[i3 + 1] = y;
      basePositions[i3 + 2] = z;
      positions[i3] = x;
      positions[i3 + 1] = y;
      positions[i3 + 2] = z;
      speeds[i] = 0.45 + Math.random() * 0.9;
    }
  }

  seedParticles();

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const pointMaterial = new THREE.PointsMaterial({
    size: 0.085,
    transparent: true,
    opacity: 0.46,
    vertexColors: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const points = new THREE.Points(geometry, pointMaterial);
  scene.add(points);

  const linePositions = new Float32Array(particleCount * 6);
  const lineColors = new Float32Array(particleCount * 6);
  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
  lineGeometry.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));

  const lineMaterial = new THREE.LineBasicMaterial({
    transparent: true,
    opacity: 0.12,
    vertexColors: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
  scene.add(lines);

  const orbGeometry = new THREE.IcosahedronGeometry(3.5, 3);
  const orbMaterial = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.045,
    wireframe: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const orb = new THREE.Mesh(orbGeometry, orbMaterial);
  orb.position.set(18, -7, -10);
  scene.add(orb);

  function applyTheme() {
    const theme = palette[getTheme()];
    pointMaterial.opacity = theme.opacity;
    lineMaterial.opacity = theme.lineOpacity;
    orbMaterial.color.copy(theme.accent);
    for (let i = 0; i < particleCount; i += 1) {
      const mix = i / Math.max(1, particleCount - 1);
      const color = theme.primary.clone().lerp(theme.secondary, (Math.sin(mix * Math.PI * 4) + 1) * 0.5);
      if (i % 7 === 0) color.lerp(theme.accent, 0.55);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    geometry.attributes.color.needsUpdate = true;
  }

  function resize() {
    width = window.innerWidth || 1;
    height = window.innerHeight || 1;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function updateLines(theme) {
    let ptr = 0;
    const maxDistance = 8.5;
    const maxDistanceSq = maxDistance * maxDistance;
    for (let i = 0; i < particleCount; i += 1) {
      let best = -1;
      let bestDist = maxDistanceSq;
      const ai = i * 3;
      for (let j = i + 1; j < particleCount; j += 1) {
        const aj = j * 3;
        const dx = positions[ai] - positions[aj];
        const dy = positions[ai + 1] - positions[aj + 1];
        const dz = positions[ai + 2] - positions[aj + 2];
        const dist = dx * dx + dy * dy + dz * dz;
        if (dist < bestDist) {
          bestDist = dist;
          best = j;
        }
      }
      if (best === -1 || ptr + 6 > linePositions.length) continue;
      const bi = best * 3;
      linePositions[ptr] = positions[ai];
      linePositions[ptr + 1] = positions[ai + 1];
      linePositions[ptr + 2] = positions[ai + 2];
      linePositions[ptr + 3] = positions[bi];
      linePositions[ptr + 4] = positions[bi + 1];
      linePositions[ptr + 5] = positions[bi + 2];

      const colorA = theme.primary.clone().lerp(theme.secondary, i / particleCount);
      const colorB = theme.secondary.clone().lerp(theme.accent, best / particleCount);
      lineColors[ptr] = colorA.r;
      lineColors[ptr + 1] = colorA.g;
      lineColors[ptr + 2] = colorA.b;
      lineColors[ptr + 3] = colorB.r;
      lineColors[ptr + 4] = colorB.g;
      lineColors[ptr + 5] = colorB.b;
      ptr += 6;
    }
    lineGeometry.setDrawRange(0, ptr / 3);
    lineGeometry.attributes.position.needsUpdate = true;
    lineGeometry.attributes.color.needsUpdate = true;
  }

  function animate() {
    const elapsed = clock.getElapsedTime();
    const theme = palette[getTheme()];
    mouse.lerp(targetMouse, 0.045);

    for (let i = 0; i < particleCount; i += 1) {
      const i3 = i * 3;
      const speed = speeds[i];
      const phase = elapsed * speed + i * 0.37;
      positions[i3] = basePositions[i3] + Math.sin(phase * 0.62) * 1.2 + mouse.x * (2.2 + (i % 5) * 0.18);
      positions[i3 + 1] = basePositions[i3 + 1] + Math.cos(phase * 0.52) * 0.9 + mouse.y * (1.6 + (i % 3) * 0.22);
      positions[i3 + 2] = basePositions[i3 + 2] + Math.sin(phase * 0.31) * 1.6;
    }

    points.rotation.z = Math.sin(elapsed * 0.05) * 0.035;
    points.rotation.x = mouse.y * 0.025;
    points.rotation.y = mouse.x * 0.035;
    orb.rotation.x = elapsed * 0.08;
    orb.rotation.y = elapsed * 0.11;
    orb.position.x = 17 + Math.sin(elapsed * 0.18) * 1.8 + mouse.x * 2;
    orb.position.y = -7 + Math.cos(elapsed * 0.15) * 1.4 + mouse.y * 1.5;

    geometry.attributes.position.needsUpdate = true;
    updateLines(theme);
    renderer.render(scene, camera);
    animationId = window.requestAnimationFrame(animate);
  }

  function onPointerMove(event) {
    targetMouse.x = (event.clientX / width - 0.5) * 2;
    targetMouse.y = -(event.clientY / height - 0.5) * 2;
  }

  const themeObserver = new MutationObserver(applyTheme);
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  window.addEventListener('resize', resize, { passive: true });
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('pagehide', () => {
    window.cancelAnimationFrame(animationId);
    renderer.dispose();
    geometry.dispose();
    lineGeometry.dispose();
    pointMaterial.dispose();
    lineMaterial.dispose();
    orbGeometry.dispose();
    orbMaterial.dispose();
  }, { once: true });

  resize();
  applyTheme();
  animate();
})();
