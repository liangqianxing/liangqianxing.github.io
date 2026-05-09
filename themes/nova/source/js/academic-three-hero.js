(function () {
  'use strict';

  const canvas = document.getElementById('academicThreeCanvas');
  const hero = document.getElementById('threeHero');
  if (!canvas || !hero || !window.THREE) return;

  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const coarsePointer = window.matchMedia && window.matchMedia('(pointer: coarse)').matches;
  if (reduceMotion || coarsePointer) return;

  const isDark = () => document.documentElement.getAttribute('data-theme') === 'dark';
  if (!isDark()) return;

  const THREE = window.THREE;
  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.6));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 1000);
  camera.position.set(0, 0, 18);

  const group = new THREE.Group();
  scene.add(group);

  const mouse = new THREE.Vector2(0, 0);
  const targetMouse = new THREE.Vector2(0, 0);
  const clock = new THREE.Clock();
  let animationId = 0;
  let width = 1;
  let height = 1;

  const particleCount = 96;
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);
  const base = [];
  const primary = new THREE.Color('#a78bfa');
  const cyan = new THREE.Color('#22d3ee');
  const green = new THREE.Color('#34d399');

  for (let i = 0; i < particleCount; i += 1) {
    const phi = Math.acos(1 - 2 * (i + 0.5) / particleCount);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    const radius = 5.2 + Math.sin(i * 1.7) * 0.45;
    const x = Math.cos(theta) * Math.sin(phi) * radius;
    const y = Math.sin(theta) * Math.sin(phi) * radius;
    const z = Math.cos(phi) * radius;
    base.push({ x, y, z, phase: Math.random() * Math.PI * 2, speed: 0.55 + Math.random() * 0.85 });
    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;
    const color = primary.clone().lerp(cyan, i / particleCount);
    if (i % 8 === 0) color.lerp(green, 0.45);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }

  const particleGeometry = new THREE.BufferGeometry();
  particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const particleMaterial = new THREE.PointsMaterial({
    size: 0.12,
    transparent: true,
    opacity: 0.92,
    vertexColors: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const particles = new THREE.Points(particleGeometry, particleMaterial);
  group.add(particles);

  const linePositions = [];
  const lineColors = [];
  for (let i = 0; i < particleCount; i += 1) {
    for (let j = i + 1; j < particleCount; j += 1) {
      const a = base[i];
      const b = base[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < 2.35) {
        linePositions.push(a.x, a.y, a.z, b.x, b.y, b.z);
        const mixA = primary.clone().lerp(cyan, i / particleCount);
        const mixB = cyan.clone().lerp(green, j / particleCount);
        lineColors.push(mixA.r, mixA.g, mixA.b, mixB.r, mixB.g, mixB.b);
      }
    }
  }

  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
  lineGeometry.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));
  const lineMaterial = new THREE.LineBasicMaterial({
    transparent: true,
    opacity: 0.2,
    vertexColors: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
  group.add(lines);

  const ringGeometry = new THREE.TorusGeometry(6.2, 0.012, 8, 160);
  const ringMaterial = new THREE.MeshBasicMaterial({
    color: '#22d3ee',
    transparent: true,
    opacity: 0.18,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const ringA = new THREE.Mesh(ringGeometry, ringMaterial);
  const ringB = new THREE.Mesh(ringGeometry, ringMaterial.clone());
  ringB.rotation.x = Math.PI / 2;
  ringB.material.color.set('#a78bfa');
  group.add(ringA, ringB);

  function resize() {
    const rect = hero.getBoundingClientRect();
    width = Math.max(1, Math.floor(rect.width));
    height = Math.max(1, Math.floor(rect.height));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function animate() {
    if (!isDark()) {
      renderer.clear();
      animationId = window.requestAnimationFrame(animate);
      return;
    }

    const time = clock.getElapsedTime();
    mouse.lerp(targetMouse, 0.055);

    for (let i = 0; i < particleCount; i += 1) {
      const p = base[i];
      const wave = Math.sin(time * p.speed + p.phase) * 0.16;
      positions[i * 3] = p.x * (1 + wave * 0.06);
      positions[i * 3 + 1] = p.y * (1 + wave * 0.06);
      positions[i * 3 + 2] = p.z * (1 + wave * 0.06);
    }
    particleGeometry.attributes.position.needsUpdate = true;

    group.rotation.y = time * 0.09 + mouse.x * 0.28;
    group.rotation.x = -0.18 + mouse.y * 0.2;
    ringA.rotation.z = time * 0.13;
    ringB.rotation.y = time * 0.11;

    renderer.render(scene, camera);
    animationId = window.requestAnimationFrame(animate);
  }

  function onPointerMove(event) {
    const rect = hero.getBoundingClientRect();
    targetMouse.x = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
    targetMouse.y = -((event.clientY - rect.top) / rect.height - 0.5) * 2;
  }

  window.addEventListener('resize', resize, { passive: true });
  hero.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('pagehide', () => {
    window.cancelAnimationFrame(animationId);
    renderer.dispose();
    particleGeometry.dispose();
    lineGeometry.dispose();
    particleMaterial.dispose();
    lineMaterial.dispose();
    ringGeometry.dispose();
    ringMaterial.dispose();
    ringB.material.dispose();
  }, { once: true });

  resize();
  animate();
})();
