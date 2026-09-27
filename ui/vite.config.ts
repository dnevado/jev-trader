import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The Python API (python -m jevbt serve) runs on :8000; the dev server proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
