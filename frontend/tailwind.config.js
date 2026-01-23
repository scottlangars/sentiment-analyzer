/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        positive: '#2ecc71',
        neutral: '#f39c12',
        negative: '#e74c3c',
      },
    },
  },
  plugins: [],
}