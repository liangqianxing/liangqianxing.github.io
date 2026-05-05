/*
 * Inspired by PavelDoGreat/WebGL-Fluid-Simulation
 * Original project: https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
 * License: MIT Copyright (c) 2017 Pavel Dobryakov
 * This file is a lightweight homepage adaptation for Nova theme.
 */
(function () {
  'use strict';

  const canvas = document.getElementById('fluidHeroCanvas');
  if (!canvas) return;

  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const coarsePointer = window.matchMedia && window.matchMedia('(pointer: coarse)').matches;
  if (reduceMotion) {
    canvas.classList.add('fluid-hero-canvas-static');
    return;
  }

  const gl = canvas.getContext('webgl', {
    alpha: true,
    depth: false,
    stencil: false,
    antialias: false,
    preserveDrawingBuffer: false,
  });
  if (!gl) {
    canvas.classList.add('fluid-hero-canvas-static');
    return;
  }

  const ext = gl.getExtension('OES_texture_float') || gl.getExtension('OES_texture_half_float');
  const linear = gl.getExtension('OES_texture_float_linear') || gl.getExtension('OES_texture_half_float_linear');
  if (!ext) {
    canvas.classList.add('fluid-hero-canvas-static');
    return;
  }

  const isHalfFloat = !!ext.HALF_FLOAT_OES;
  const textureType = isHalfFloat ? ext.HALF_FLOAT_OES : gl.FLOAT;
  const filter = linear ? gl.LINEAR : gl.NEAREST;

  const config = {
    simResolution: coarsePointer ? 64 : 96,
    dyeResolution: coarsePointer ? 256 : 512,
    densityDissipation: 0.992,
    velocityDissipation: 0.975,
    pressureDissipation: 0.82,
    pressureIterations: coarsePointer ? 8 : 12,
    curl: 28,
    splatRadius: coarsePointer ? 0.022 : 0.018,
  };

  const baseVertex = `
    precision highp float;
    attribute vec2 aPosition;
    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform vec2 texelSize;
    void main () {
      vUv = aPosition * 0.5 + 0.5;
      vL = vUv - vec2(texelSize.x, 0.0);
      vR = vUv + vec2(texelSize.x, 0.0);
      vT = vUv + vec2(0.0, texelSize.y);
      vB = vUv - vec2(0.0, texelSize.y);
      gl_Position = vec4(aPosition, 0.0, 1.0);
    }
  `;

  const clearShader = `
    precision mediump float;
    varying highp vec2 vUv;
    uniform sampler2D uTexture;
    uniform float value;
    void main () { gl_FragColor = value * texture2D(uTexture, vUv); }
  `;

  const displayShader = `
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uTexture;
    void main () {
      vec3 c = texture2D(uTexture, vUv).rgb;
      float a = smoothstep(0.02, 0.55, max(max(c.r, c.g), c.b));
      c = pow(c, vec3(0.92));
      gl_FragColor = vec4(c, a * 0.92);
    }
  `;

  const splatShader = `
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uTarget;
    uniform float aspectRatio;
    uniform vec3 color;
    uniform vec2 point;
    uniform float radius;
    void main () {
      vec2 p = vUv - point.xy;
      p.x *= aspectRatio;
      vec3 splat = exp(-dot(p, p) / radius) * color;
      vec3 base = texture2D(uTarget, vUv).xyz;
      gl_FragColor = vec4(base + splat, 1.0);
    }
  `;

  const advectionShader = `
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uVelocity;
    uniform sampler2D uSource;
    uniform vec2 texelSize;
    uniform float dt;
    uniform float dissipation;
    void main () {
      vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
      gl_FragColor = dissipation * texture2D(uSource, coord);
    }
  `;

  const divergenceShader = `
    precision mediump float;
    varying highp vec2 vUv;
    varying highp vec2 vL;
    varying highp vec2 vR;
    varying highp vec2 vT;
    varying highp vec2 vB;
    uniform sampler2D uVelocity;
    void main () {
      float L = texture2D(uVelocity, vL).x;
      float R = texture2D(uVelocity, vR).x;
      float T = texture2D(uVelocity, vT).y;
      float B = texture2D(uVelocity, vB).y;
      vec2 C = texture2D(uVelocity, vUv).xy;
      if (vL.x < 0.0) L = -C.x;
      if (vR.x > 1.0) R = -C.x;
      if (vT.y > 1.0) T = -C.y;
      if (vB.y < 0.0) B = -C.y;
      float div = 0.5 * (R - L + T - B);
      gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
    }
  `;

  const curlShader = `
    precision mediump float;
    varying highp vec2 vUv;
    varying highp vec2 vL;
    varying highp vec2 vR;
    varying highp vec2 vT;
    varying highp vec2 vB;
    uniform sampler2D uVelocity;
    void main () {
      float L = texture2D(uVelocity, vL).y;
      float R = texture2D(uVelocity, vR).y;
      float T = texture2D(uVelocity, vT).x;
      float B = texture2D(uVelocity, vB).x;
      float vorticity = R - L - T + B;
      gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
    }
  `;

  const vorticityShader = `
    precision highp float;
    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uVelocity;
    uniform sampler2D uCurl;
    uniform float curl;
    uniform float dt;
    void main () {
      float L = texture2D(uCurl, vL).x;
      float R = texture2D(uCurl, vR).x;
      float T = texture2D(uCurl, vT).x;
      float B = texture2D(uCurl, vB).x;
      float C = texture2D(uCurl, vUv).x;
      vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
      force /= length(force) + 0.0001;
      force *= curl * C;
      force.y *= -1.0;
      vec2 velocity = texture2D(uVelocity, vUv).xy;
      velocity += force * dt;
      gl_FragColor = vec4(velocity, 0.0, 1.0);
    }
  `;

  const pressureShader = `
    precision mediump float;
    varying highp vec2 vUv;
    varying highp vec2 vL;
    varying highp vec2 vR;
    varying highp vec2 vT;
    varying highp vec2 vB;
    uniform sampler2D uPressure;
    uniform sampler2D uDivergence;
    void main () {
      float L = texture2D(uPressure, vL).x;
      float R = texture2D(uPressure, vR).x;
      float T = texture2D(uPressure, vT).x;
      float B = texture2D(uPressure, vB).x;
      float C = texture2D(uPressure, vUv).x;
      if (vL.x < 0.0) L = C;
      if (vR.x > 1.0) R = C;
      if (vT.y > 1.0) T = C;
      if (vB.y < 0.0) B = C;
      float divergence = texture2D(uDivergence, vUv).x;
      float pressure = (L + R + B + T - divergence) * 0.25;
      gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
    }
  `;

  const gradientSubtractShader = `
    precision mediump float;
    varying highp vec2 vUv;
    varying highp vec2 vL;
    varying highp vec2 vR;
    varying highp vec2 vT;
    varying highp vec2 vB;
    uniform sampler2D uPressure;
    uniform sampler2D uVelocity;
    void main () {
      float L = texture2D(uPressure, vL).x;
      float R = texture2D(uPressure, vR).x;
      float T = texture2D(uPressure, vT).x;
      float B = texture2D(uPressure, vB).x;
      vec2 velocity = texture2D(uVelocity, vUv).xy;
      velocity.xy -= vec2(R - L, T - B);
      gl_FragColor = vec4(velocity, 0.0, 1.0);
    }
  `;

  const blit = (() => {
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
    return (target, clear) => {
      if (target == null) {
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      } else {
        gl.viewport(0, 0, target.width, target.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      }
      if (clear) gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    };
  })();

  function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  }

  function createProgram(fragmentSource) {
    const program = gl.createProgram();
    gl.attachShader(program, compileShader(gl.VERTEX_SHADER, baseVertex));
    gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentSource));
    gl.bindAttribLocation(program, 0, 'aPosition');
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    const uniforms = {};
    const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const name = gl.getActiveUniform(program, i).name;
      uniforms[name] = gl.getUniformLocation(program, name);
    }
    return { program, uniforms, bind: () => gl.useProgram(program) };
  }

  function createFBO(width, height, internalFormat, format, param) {
    gl.activeTexture(gl.TEXTURE0);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, textureType, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, width, height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return { texture, fbo, width, height, texelSizeX: 1 / width, texelSizeY: 1 / height, attach: id => (gl.activeTexture(gl.TEXTURE0 + id), gl.bindTexture(gl.TEXTURE_2D, texture), id) };
  }

  function createDoubleFBO(width, height, internalFormat, format, param) {
    let fbo1 = createFBO(width, height, internalFormat, format, param);
    let fbo2 = createFBO(width, height, internalFormat, format, param);
    return { width, height, texelSizeX: 1 / width, texelSizeY: 1 / height, get read() { return fbo1; }, set read(v) { fbo1 = v; }, get write() { return fbo2; }, set write(v) { fbo2 = v; }, swap() { const temp = fbo1; fbo1 = fbo2; fbo2 = temp; } };
  }

  const programs = {
    clear: createProgram(clearShader),
    display: createProgram(displayShader),
    splat: createProgram(splatShader),
    advection: createProgram(advectionShader),
    divergence: createProgram(divergenceShader),
    curl: createProgram(curlShader),
    vorticity: createProgram(vorticityShader),
    pressure: createProgram(pressureShader),
    gradientSubtract: createProgram(gradientSubtractShader),
  };

  let dye, velocity, divergence, curl, pressure;
  let lastTime = Date.now();
  let colorIndex = 0;
  const palette = [
    [0.10, 0.85, 1.00],
    [0.12, 0.92, 0.18],
    [0.95, 0.08, 0.82],
    [1.00, 0.82, 0.05],
    [0.15, 0.12, 1.00],
    [1.00, 0.12, 0.06],
    [0.00, 0.70, 0.55],
    [0.70, 0.08, 1.00],
  ];

  function getResolution(resolution) {
    const aspect = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspect < 1) return { width: resolution, height: Math.round(resolution / aspect) };
    return { width: Math.round(resolution * aspect), height: resolution };
  }

  function initFramebuffers() {
    const sim = getResolution(config.simResolution);
    const dyeRes = getResolution(config.dyeResolution);
    velocity = createDoubleFBO(sim.width, sim.height, gl.RGBA, gl.RGBA, filter);
    dye = createDoubleFBO(dyeRes.width, dyeRes.height, gl.RGBA, gl.RGBA, filter);
    divergence = createFBO(sim.width, sim.height, gl.RGBA, gl.RGBA, gl.NEAREST);
    curl = createFBO(sim.width, sim.height, gl.RGBA, gl.RGBA, gl.NEAREST);
    pressure = createDoubleFBO(sim.width, sim.height, gl.RGBA, gl.RGBA, gl.NEAREST);
  }

  function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      initFramebuffers();
      seed();
    }
  }

  function splat(x, y, dx, dy, color) {
    const aspectRatio = canvas.width / canvas.height;
    programs.splat.bind();
    gl.uniform1i(programs.splat.uniforms.uTarget, velocity.read.attach(0));
    gl.uniform1f(programs.splat.uniforms.aspectRatio, aspectRatio);
    gl.uniform2f(programs.splat.uniforms.point, x, y);
    gl.uniform3f(programs.splat.uniforms.color, dx, dy, 0.0);
    gl.uniform1f(programs.splat.uniforms.radius, config.splatRadius);
    blit(velocity.write);
    velocity.swap();

    gl.uniform1i(programs.splat.uniforms.uTarget, dye.read.attach(0));
    gl.uniform3f(programs.splat.uniforms.color, color[0], color[1], color[2]);
    gl.uniform1f(programs.splat.uniforms.radius, config.splatRadius * 1.25);
    blit(dye.write);
    dye.swap();
  }

  function seed() {
    for (let i = 0; i < 18; i++) {
      const x = 0.18 + Math.random() * 0.64;
      const y = 0.16 + Math.random() * 0.68;
      const c = palette[i % palette.length];
      splat(x, y, (Math.random() - 0.5) * 1450, (Math.random() - 0.5) * 1450, c);
    }
  }

  function step(dt) {
    gl.disable(gl.BLEND);

    programs.curl.bind();
    gl.uniform2f(programs.curl.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(programs.curl.uniforms.uVelocity, velocity.read.attach(0));
    blit(curl);

    programs.vorticity.bind();
    gl.uniform2f(programs.vorticity.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(programs.vorticity.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(programs.vorticity.uniforms.uCurl, curl.attach(1));
    gl.uniform1f(programs.vorticity.uniforms.curl, config.curl);
    gl.uniform1f(programs.vorticity.uniforms.dt, dt);
    blit(velocity.write);
    velocity.swap();

    programs.divergence.bind();
    gl.uniform2f(programs.divergence.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(programs.divergence.uniforms.uVelocity, velocity.read.attach(0));
    blit(divergence);

    programs.clear.bind();
    gl.uniform1i(programs.clear.uniforms.uTexture, pressure.read.attach(0));
    gl.uniform1f(programs.clear.uniforms.value, config.pressureDissipation);
    blit(pressure.write);
    pressure.swap();

    programs.pressure.bind();
    gl.uniform2f(programs.pressure.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(programs.pressure.uniforms.uDivergence, divergence.attach(0));
    for (let i = 0; i < config.pressureIterations; i++) {
      gl.uniform1i(programs.pressure.uniforms.uPressure, pressure.read.attach(1));
      blit(pressure.write);
      pressure.swap();
    }

    programs.gradientSubtract.bind();
    gl.uniform2f(programs.gradientSubtract.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(programs.gradientSubtract.uniforms.uPressure, pressure.read.attach(0));
    gl.uniform1i(programs.gradientSubtract.uniforms.uVelocity, velocity.read.attach(1));
    blit(velocity.write);
    velocity.swap();

    programs.advection.bind();
    gl.uniform2f(programs.advection.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(programs.advection.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(programs.advection.uniforms.uSource, velocity.read.attach(0));
    gl.uniform1f(programs.advection.uniforms.dt, dt);
    gl.uniform1f(programs.advection.uniforms.dissipation, config.velocityDissipation);
    blit(velocity.write);
    velocity.swap();

    gl.uniform2f(programs.advection.uniforms.texelSize, dye.texelSizeX, dye.texelSizeY);
    gl.uniform1i(programs.advection.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(programs.advection.uniforms.uSource, dye.read.attach(1));
    gl.uniform1f(programs.advection.uniforms.dissipation, config.densityDissipation);
    blit(dye.write);
    dye.swap();
  }

  function render() {
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.BLEND);
    programs.display.bind();
    gl.uniform1i(programs.display.uniforms.uTexture, dye.read.attach(0));
    blit(null, true);
  }

  function update() {
    resizeCanvas();
    const now = Date.now();
    const dt = Math.min((now - lastTime) / 1000, 0.0166);
    lastTime = now;
    step(dt);
    render();
    requestAnimationFrame(update);
  }

  let lastPointer = null;
  function pointerMove(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const x = (clientX - rect.left) / rect.width;
    const y = 1 - (clientY - rect.top) / rect.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    const prev = lastPointer || { x, y };
    const dx = (x - prev.x) * 9000;
    const dy = (y - prev.y) * 9000;
    const color = palette[colorIndex++ % palette.length];
    splat(x, y, dx, dy, color);
    lastPointer = { x, y };
  }

  canvas.addEventListener('pointermove', event => pointerMove(event.clientX, event.clientY), { passive: true });
  canvas.addEventListener('pointerleave', () => { lastPointer = null; }, { passive: true });
  canvas.addEventListener('pointerdown', event => {
    lastPointer = null;
    pointerMove(event.clientX, event.clientY);
  }, { passive: true });

  setInterval(() => {
    const c = palette[colorIndex++ % palette.length];
    splat(0.08 + Math.random() * 0.84, 0.10 + Math.random() * 0.78, (Math.random() - 0.5) * 1800, (Math.random() - 0.5) * 1800, c);
  }, coarsePointer ? 1800 : 980);

  try {
    resizeCanvas();
    seed();
    update();
  } catch (error) {
    canvas.classList.add('fluid-hero-canvas-static');
    console.warn('Fluid hero disabled:', error);
  }
})();
