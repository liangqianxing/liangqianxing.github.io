<template>
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path v-for="(path, index) in paths[name] || paths.code" :key="index" :d="path" /></svg>
</template>
<script setup lang="ts">
defineProps<{ name: string }>()
const paths: Record<string, string[]> = {
  book: ['M12 5.5C8.5 3.5 5.5 3 2 4v15c3.5-1 6.5-.5 10 1.5 3.5-2 6.5-2.5 10-1.5V4c-3.5-1-6.5-.5-10 1.5Z', 'M12 5.5v15M5.5 8l3 .7M5.5 11.5l3 .7M15.5 8.7l3-.7M15.5 12.2l3-.7'],
  research: ['m2 9 10-5 10 5-10 5L2 9Z', 'M6 11v6c4 3 8 3 12 0v-6M22 9v8'],
  layers: ['m12 3 10 5-10 5L2 8l10-5Z', 'm2 12 10 5 10-5M2 16l10 5 10-5'],
  code: ['m8 6-6 6 6 6M16 6l6 6-6 6M14 3l-4 18'],
  terminal: ['M4 3h16a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z', 'm7 8 4 4-4 4M14 16h3'],
  archive: ['M3 3h18v5H3V3ZM5 8v13h14V8M9 12h6'],
  friends: ['M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M16 4a4 4 0 0 1 0 8M22 21v-2a4 4 0 0 0-3-3.87', 'M13 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z'],
  person: ['M16 6a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z', 'M4 22v-3a8 8 0 0 1 16 0v3'],
  paper: ['M14 2H5v20h14V7l-5-5ZM14 2v5h5M8 12h8M8 16h8'],
  external: ['M14 3h7v7M21 3 10 14M10 3H3v18h18v-7'],
  github: ['M9 19c-4.3 1.3-4.3-2.2-6-2.7M15 22v-3.8a3.3 3.3 0 0 0-.9-2.6c3-.3 6.2-1.5 6.2-6.8a5.4 5.4 0 0 0-1.4-3.7 5 5 0 0 0-.1-3.7s-1.2-.4-3.8 1.4a13.2 13.2 0 0 0-7 0C5.4 1 4.2 1.4 4.2 1.4a5 5 0 0 0-.1 3.7 5.4 5.4 0 0 0-1.4 3.7c0 5.3 3.2 6.5 6.2 6.8a3.3 3.3 0 0 0-.9 2.6V22'],
  'arrow-down': ['M12 4v16m-6-6 6 6 6-6'], 'arrow-up': ['M12 20V4m-6 6 6-6 6 6'], 'arrow-left': ['M20 12H4m6-6-6 6 6 6'],
}
</script>
