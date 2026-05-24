import typography from '@tailwindcss/typography';

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,ts}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        accent: {
          DEFAULT: '#534AB7',
          light: '#AFA9EC',
          dark: '#3C3489',
        },
      },
      fontFamily: {
        sans: [
          'Inter',
          'PingFang SC',
          'Noto Sans SC',
          'Hiragino Sans GB',
          'Microsoft YaHei',
          'ui-sans-serif',
          'system-ui',
          'sans-serif',
        ],
        mono: [
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'Consolas',
          'Liberation Mono',
          'monospace',
        ],
      },
    },
  },
  plugins: [typography],
};
