/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          yellow: "#F2C24E",
          yellowSoft: "#F7D98A",
        },
      },
    },
  },
  plugins: [
      require('tailwind-scrollbar'),
  ],
};
