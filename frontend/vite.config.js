import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import basicSsl from "@vitejs/plugin-basic-ssl";

export default defineConfig({
  plugins: [react(), basicSsl()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setupTests.js",
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    https: true,
    proxy: {
      "/api": {
        target: "https://localhost:5000",
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
