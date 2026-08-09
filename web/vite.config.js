import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // onnxruntime-web ships prebuilt wasm; don't let Vite try to optimize it
  optimizeDeps: { exclude: ['onnxruntime-web'] },
})
