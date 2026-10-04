<template>
  <aside class="notebook-code-canvas" aria-label="作者的代码名片">
    <div class="notebook-code-glow" aria-hidden="true" />
    <div class="notebook-code-backing" aria-hidden="true" />
    <svg class="notebook-code-doodle" width="200" height="100" viewBox="0 0 200 100" fill="none" aria-hidden="true"><path d="M10 80C40 70 80 90 120 70C160 50 180 20 190 10" stroke="currentColor" stroke-width="2" stroke-dasharray="6 6" /></svg>
    <div class="notebook-code-window">
      <span class="notebook-code-note" aria-hidden="true">&lt;/&gt;</span>
      <div class="notebook-code-header">
        <span class="notebook-code-dots" aria-hidden="true"><i /><i /><i /></span>
        <span>profile.ts</span>
      </div>
      <pre class="notebook-code-profile"><code><span class="notebook-code-keyword">const</span> <span class="notebook-code-variable">enhao</span> = {
  name: <span class="notebook-code-string">"{{ name }}"</span>,
  notebook: <span class="notebook-code-string">"{{ notebook }}"</span>,
  focus: [<span class="notebook-code-string">"LLM"</span>, <span class="notebook-code-string">"Agent"</span>, <span class="notebook-code-string">"Systems"</span>],
  next: <span class="notebook-code-string">"ECNU · 2027 incoming"</span>
};</code></pre>
      <span class="notebook-code-logo" aria-hidden="true"><img src="/logo.svg?v=3" alt="" width="48" height="48" /></span>
    </div>
    <svg class="notebook-code-star" width="44" height="44" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2 14.5 9.5 22 12 14.5 14.5 12 22 9.5 14.5 2 12 9.5 9.5Z" /></svg>
    <span class="notebook-code-dot" aria-hidden="true" />
    <span class="notebook-code-sticker" aria-hidden="true">LEARN · BUILD · REPEAT</span>
  </aside>
</template>

<script setup lang="ts">
defineProps<{ name: string; notebook: string }>()
</script>

<style scoped>
.notebook-code-canvas {
  --syntax-keyword: #7550c0;
  --syntax-variable: #c53c66;
  --syntax-string: #08785b;
  --note-bg: #ffe898;
  --note-ink: #b45309;
  --sticker-bg: #f5ebff;
  --sticker-ink: #7550aa;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 0;
  margin-inline: 12px 18px;
}
.notebook-code-glow,
.notebook-code-backing,
.notebook-code-doodle,
.notebook-code-note,
.notebook-code-logo,
.notebook-code-star,
.notebook-code-dot,
.notebook-code-sticker { pointer-events: none; }
.notebook-code-glow {
  position: absolute;
  width: 360px;
  max-width: 110%;
  height: 300px;
  background: radial-gradient(ellipse at 60% 30%, #b6a4ff80, transparent 68%), radial-gradient(ellipse at 20% 80%, #9ccfff90, transparent 70%);
  filter: blur(22px);
}
.notebook-code-backing {
  position: absolute;
  inset: 10px -10px -12px 12px;
  z-index: 1;
  border: 1px solid color-mix(in srgb, var(--accent) 16%, transparent);
  border-radius: 20px;
  background: color-mix(in srgb, var(--accent) 24%, var(--surface));
  transform: rotate(3deg);
}
.notebook-code-doodle { position: absolute; top: -38px; left: -36px; z-index: 2; color: var(--accent); opacity: .6; transform: rotate(-12deg); }
.notebook-code-window {
  position: relative;
  z-index: 5;
  width: 100%;
  min-height: 244px;
  padding: 22px;
  border: 1px solid color-mix(in srgb, var(--accent) 12%, var(--surface));
  border-radius: 20px;
  background: var(--surface);
  color: var(--text);
  box-shadow: 0 15px 35px #0000000f;
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.85;
}
.notebook-code-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid var(--line); color: var(--muted); font-size: 11px; }
.notebook-code-dots { display: flex; gap: 6px; }
.notebook-code-dots i { width: 10px; height: 10px; border-radius: 50%; background: #ff5f56; }
.notebook-code-dots i:nth-child(2) { background: #ffbd2e; }
.notebook-code-dots i:nth-child(3) { background: #27c93f; }
.notebook-code-profile { margin: 0; padding: 0; border: 0; background: transparent; color: var(--text); font: inherit; white-space: pre-wrap; overflow-wrap: anywhere; }
.notebook-code-profile code { display: block; padding: 0; background: transparent; color: inherit; font: inherit; }
.notebook-code-keyword { color: var(--syntax-keyword); }
.notebook-code-variable { color: var(--syntax-variable); }
.notebook-code-string { color: var(--syntax-string); }
.notebook-code-note { position: absolute; top: -38px; right: 12px; z-index: 7; display: grid; place-items: center; width: 58px; height: 58px; border-radius: 4px; background: var(--note-bg); color: var(--note-ink); box-shadow: 2px 4px 10px #00000014; transform: rotate(8deg); font-size: 22px; }
.notebook-code-logo { position: absolute; right: 14px; bottom: -26px; z-index: 7; width: 64px; height: 64px; padding: 8px; border: 1px solid color-mix(in srgb, var(--note-bg) 45%, var(--line)); border-radius: 16px; background: var(--surface); box-shadow: 0 10px 25px #00000019; transform: rotate(-6deg); }
.notebook-code-logo img { width: 100%; height: 100%; }
.notebook-code-star { position: absolute; top: -29px; left: -24px; z-index: 6; color: #f7b827; transform: rotate(8deg); filter: drop-shadow(0 2px 4px #fbbf2433); }
.notebook-code-dot { position: absolute; bottom: 24%; left: -17px; z-index: 6; width: 12px; height: 12px; border-radius: 50%; background: #ff9387; }
.notebook-code-sticker { position: absolute; bottom: -14px; left: 10px; z-index: 7; padding: 7px 11px; border: 1px solid color-mix(in srgb, var(--sticker-ink) 18%, var(--surface)); border-radius: 8px; background: var(--sticker-bg); color: var(--sticker-ink); transform: rotate(-4deg); font: 10px var(--font-mono); letter-spacing: .06em; }
:global(html.dark .notebook-code-canvas) { --syntax-keyword: #c9afff; --syntax-variable: #ff9bb9; --syntax-string: #86e7c5; --note-bg: #d4a847; --note-ink: #382604; --sticker-bg: #37294e; --sticker-ink: #ddc1ff; }
:global(html.cyber .notebook-code-canvas) { --syntax-keyword: #81e5f2; --syntax-variable: #ffa9c7; --syntax-string: #9affc4; --note-bg: #d4a847; --note-ink: #382604; --sticker-bg: #193b32; --sticker-ink: #9affc4; }
@media (max-width: 768px) {
  .notebook-code-canvas { margin-inline: 12px 16px; }
  .notebook-code-window { min-height: 0; padding: 16px; font-size: 12px; line-height: 1.7; }
  .notebook-code-header { margin-bottom: 12px; padding-bottom: 10px; }
  .notebook-code-glow { width: 100%; height: 240px; }
  .notebook-code-star { top: -26px; left: -18px; width: 36px; height: 36px; }
  .notebook-code-note { top: -30px; right: 16px; width: 44px; height: 44px; font-size: 18px; }
  .notebook-code-logo { right: 4px; bottom: -22px; width: 48px; height: 48px; padding: 6px; }
  .notebook-code-doodle { display: none; }
  .notebook-code-sticker { left: 2px; font-size: 9px; }
}
</style>
