(function () {
  const canvas = document.getElementById('threePlaygroundCanvas');
  const root = document.getElementById('threePlayground');
  const shuffleButton = document.getElementById('threePlaygroundShuffle');
  if (!canvas || !root) return;

  const gl = canvas.getContext('webgl', { alpha: true, antialias: false, powerPreference: 'high-performance' });
  if (!gl) {
    root.classList.add('three-playground-fallback');
    return;
  }

  const vertexSource = `
    attribute vec3 aPosition;
    attribute vec3 aColor;
    attribute float aSize;
    attribute float aSeed;
    attribute float aKind;
    uniform float uTime;
    uniform vec2 uPointer;
    uniform vec2 uBurst0;
    uniform vec2 uBurst1;
    uniform vec2 uBurst2;
    uniform float uPower0;
    uniform float uPower1;
    uniform float uPower2;
    uniform float uMode;
    uniform float uZoom;
    uniform float uPixelRatio;
    varying vec3 vColor;
    varying float vAlpha;
    varying float vSpark;

    mat2 rotate2d(float a) {
      float s = sin(a);
      float c = cos(a);
      return mat2(c, -s, s, c);
    }

    float burstWave(inout vec3 p, vec2 center, float power, float speed, float width, float strength) {
      vec2 world = center * vec2(6.4, 3.7);
      vec2 delta = p.xy - world;
      float dist = length(delta);
      float radius = (1.0 - power) * speed;
      float ring = exp(-pow(dist - radius, 2.0) * width) * power;
      p.xy += normalize(delta + vec2(0.001)) * ring * strength;
      p.z += ring * 1.3;
      return ring;
    }

    void main() {
      vec3 p = aPosition;
      float r = max(length(p.xy), 0.08);
      float swirl = uTime * (0.08 + 0.22 / (r + 1.0)) + aSeed * 0.08;
      p.xy = rotate2d(swirl) * p.xy;

      vec2 pointer = uPointer * vec2(6.4, 3.7);
      vec2 pull = pointer - p.xy;
      float pullDist = max(length(pull), 0.24);
      p.xy += normalize(pull) * (0.08 + uMode * 0.07) / pullDist;

      float spark = 0.0;
      spark += burstWave(p, uBurst0, uPower0, 8.2, 2.8, 1.15 + uMode * 0.42);
      spark += burstWave(p, uBurst1, uPower1, 6.2, 3.8, 0.92 + uMode * 0.36);
      spark += burstWave(p, uBurst2, uPower2, 10.0, 1.9, 1.42 + uMode * 0.55);

      float ribbon = sin(r * 1.6 - uTime * (0.7 + uMode * 0.25) + aSeed) * 0.18;
      p.z += ribbon + spark * 1.1;

      float perspective = 1.0 / (1.0 + (p.z + uZoom) * 0.045);
      vec2 projected = p.xy * 0.112 * perspective;
      gl_Position = vec4(projected, p.z * 0.014, 1.0);
      gl_PointSize = aSize * uPixelRatio * (20.0 + spark * 46.0 + aKind * 8.0) * perspective;

      vec3 hot = mix(vec3(0.30, 0.94, 1.0), vec3(1.0, 0.82, 0.28), spark);
      vColor = mix(aColor, hot, clamp(spark * 0.88, 0.0, 1.0));
      vAlpha = 0.58 + aKind * 0.34 + spark * 0.35;
      vSpark = spark;
    }
  `;

  const fragmentSource = `
    precision mediump float;
    varying vec3 vColor;
    varying float vAlpha;
    varying float vSpark;
    void main() {
      vec2 uv = gl_PointCoord - vec2(0.5);
      float d = dot(uv, uv);
      if (d > 0.25) discard;
      float core = smoothstep(0.18, 0.006, d);
      float halo = smoothstep(0.25, 0.04, d) * (0.18 + vSpark * 0.55);
      gl_FragColor = vec4(vColor, (core + halo) * vAlpha);
    }
  `;

  function shader(type, source) {
    const item = gl.createShader(type);
    gl.shaderSource(item, source);
    gl.compileShader(item);
    if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) {
      console.warn(gl.getShaderInfoLog(item));
      gl.deleteShader(item);
      return null;
    }
    return item;
  }

  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return;

  gl.useProgram(program);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  gl.disable(gl.DEPTH_TEST);

  const palettes = [
    [[0.20, 0.85, 1.00], [0.60, 0.44, 1.00], [1.00, 0.34, 0.72], [1.00, 0.78, 0.28]],
    [[0.18, 1.00, 0.72], [0.38, 0.68, 1.00], [0.90, 0.54, 1.00], [1.00, 0.92, 0.36]],
    [[1.00, 0.38, 0.18], [1.00, 0.78, 0.22], [0.25, 0.92, 1.00], [0.96, 0.40, 0.86]]
  ];
  let paletteIndex = 0;
  let mode = 0;
  let palette = palettes[paletteIndex];

  const count = window.innerWidth < 720 ? 2800 : 6200;
  const stride = 9;
  const data = new Float32Array(count * stride);

  function writeParticle(i, x, y, z, color, size, seed, kind) {
    const base = i * stride;
    data[base] = x;
    data[base + 1] = y;
    data[base + 2] = z;
    data[base + 3] = color[0];
    data[base + 4] = color[1];
    data[base + 5] = color[2];
    data[base + 6] = size;
    data[base + 7] = seed;
    data[base + 8] = kind;
  }

  function fillData() {
    for (let i = 0; i < count; i++) {
      const ribbon = i < count * 0.76;
      const sparks = !ribbon && i < count * 0.9;
      if (ribbon) {
        const lane = i % 5;
        const radius = 0.9 + Math.pow(Math.random(), 0.58) * 9.8;
        const angle = Math.random() * Math.PI * 2 + radius * (0.38 + lane * 0.045);
        const wave = Math.sin(radius * 1.3 + lane) * 0.42;
        const x = Math.cos(angle) * (radius + wave);
        const y = Math.sin(angle) * (radius + wave) * (0.42 + lane * 0.035);
        const z = (Math.random() - 0.5) * (2.8 + radius * 0.22);
        writeParticle(i, x, y, z, palette[(i + lane) % palette.length], 0.30 + Math.random() * 0.66, Math.random() * 6.28, 0.95);
      } else if (sparks) {
        const radius = Math.pow(Math.random(), 0.45) * 8.5;
        const angle = Math.random() * Math.PI * 2;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius * 0.72;
        const z = (Math.random() - 0.5) * 7;
        writeParticle(i, x, y, z, palette[(i + 2) % palette.length], 0.18 + Math.random() * 0.42, Math.random() * 6.28, 0.58);
      } else {
        const radius = 7 + Math.random() * 11;
        const angle = Math.random() * Math.PI * 2;
        writeParticle(i, Math.cos(angle) * radius, Math.sin(angle) * radius * 0.72, (Math.random() - 0.5) * 13, [0.62, 0.78, 1.0], 0.12 + Math.random() * 0.28, Math.random() * 6.28, 0.22);
      }
    }
  }

  fillData();
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);

  function attrib(name, size, offset) {
    const location = gl.getAttribLocation(program, name);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, stride * 4, offset * 4);
  }
  attrib('aPosition', 3, 0);
  attrib('aColor', 3, 3);
  attrib('aSize', 1, 6);
  attrib('aSeed', 1, 7);
  attrib('aKind', 1, 8);

  const uniforms = {
    time: gl.getUniformLocation(program, 'uTime'),
    pointer: gl.getUniformLocation(program, 'uPointer'),
    bursts: [
      gl.getUniformLocation(program, 'uBurst0'),
      gl.getUniformLocation(program, 'uBurst1'),
      gl.getUniformLocation(program, 'uBurst2')
    ],
    powers: [
      gl.getUniformLocation(program, 'uPower0'),
      gl.getUniformLocation(program, 'uPower1'),
      gl.getUniformLocation(program, 'uPower2')
    ],
    mode: gl.getUniformLocation(program, 'uMode'),
    zoom: gl.getUniformLocation(program, 'uZoom'),
    pixelRatio: gl.getUniformLocation(program, 'uPixelRatio')
  };

  const pointer = { x: 0.25, y: -0.05 };
  const target = { x: 0.25, y: -0.05 };
  const bursts = [
    { x: -0.2, y: 0.1, power: 0.0 },
    { x: 0.35, y: -0.2, power: 0.0 },
    { x: 0.0, y: 0.0, power: 0.0 }
  ];
  let burstCursor = 0;
  let zoom = 0;
  let time = 0;
  let frame = 0;

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = root.clientWidth || window.innerWidth;
    const height = root.clientHeight || window.innerHeight;
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(height * dpr));
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.uniform1f(uniforms.pixelRatio, dpr);
  }

  function readPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width - 0.5) * 2,
      y: -((event.clientY - rect.top) / rect.height - 0.5) * 2
    };
  }

  function addBurst(point, power) {
    const item = bursts[burstCursor];
    item.x = point.x;
    item.y = point.y;
    item.power = power;
    burstCursor = (burstCursor + 1) % bursts.length;
    root.classList.remove('three-playground-pop');
    void root.offsetWidth;
    root.classList.add('three-playground-pop');
  }

  window.addEventListener('resize', resize, { passive: true });
  window.addEventListener('pointermove', (event) => {
    const point = readPoint(event);
    target.x = point.x;
    target.y = point.y;
  }, { passive: true });
  window.addEventListener('pointerdown', (event) => {
    const point = readPoint(event);
    target.x = point.x;
    target.y = point.y;
    addBurst(point, event.shiftKey ? 1.8 : 1.15);
  }, { passive: true });
  window.addEventListener('dblclick', (event) => {
    const point = readPoint(event);
    addBurst(point, 2.2);
  }, { passive: true });
  window.addEventListener('wheel', (event) => {
    zoom = Math.max(-2.5, Math.min(3.8, zoom + event.deltaY * 0.002));
  }, { passive: true });

  shuffleButton?.addEventListener('click', (event) => {
    event.stopPropagation();
    paletteIndex = (paletteIndex + 1) % palettes.length;
    mode = (mode + 1) % 3;
    palette = palettes[paletteIndex];
    fillData();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    addBurst({ x: 0.45, y: -0.15 }, 1.35);
    shuffleButton.textContent = ['切换：星河', '切换：极光', '切换：霓虹'][mode];
  });

  function render() {
    frame = requestAnimationFrame(render);
    time += 0.016;
    pointer.x += (target.x - pointer.x) * 0.075;
    pointer.y += (target.y - pointer.y) * 0.075;
    bursts.forEach((item) => { item.power *= 0.965; });

    gl.clearColor(0.006, 0.008, 0.022, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform1f(uniforms.time, time);
    gl.uniform2f(uniforms.pointer, pointer.x, pointer.y);
    for (let i = 0; i < bursts.length; i++) {
      gl.uniform2f(uniforms.bursts[i], bursts[i].x, bursts[i].y);
      gl.uniform1f(uniforms.powers[i], bursts[i].power);
    }
    gl.uniform1f(uniforms.mode, mode);
    gl.uniform1f(uniforms.zoom, zoom);
    gl.drawArrays(gl.POINTS, 0, count);
  }

  resize();
  render();

  window.addEventListener('pagehide', () => {
    cancelAnimationFrame(frame);
    gl.deleteBuffer(buffer);
    gl.deleteProgram(program);
  });
})();
