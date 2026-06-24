import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// VITE_API_BASE points at the FastAPI orchestrator (default localhost:8080).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: true },
});
