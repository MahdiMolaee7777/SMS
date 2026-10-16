/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './templates/**/*.html',
    './detector/templates/**/*.html',
    './assets/js/**/*.js',
  ],
  theme: {
    extend: {
      colors: {
        // Dark navy "cybersecurity" surface palette.
        ink: {
          950: '#060a15',
          900: '#0b1120',
          800: '#131c31',
          700: '#1c2740',
        },
        brand: {
          100: '#dceeff',
          200: '#b6dfff',
          300: '#7cc7ff',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
        },
      },
      boxShadow: {
        card: '0 12px 32px -18px rgba(2, 6, 23, 0.9)',
        glow: '0 10px 30px -12px rgba(56, 189, 248, 0.55)',
        'glow-danger': '0 12px 36px -18px rgba(244, 63, 94, 0.65)',
        'glow-success': '0 12px 36px -18px rgba(16, 185, 129, 0.55)',
      },
      transitionTimingFunction: {
        'out-quint': 'cubic-bezier(0.22, 1, 0.36, 1)',
      },
    },
  },
  plugins: [],
};
