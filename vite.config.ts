import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy'

// https://vitejs.dev/config/
export default defineConfig({
	base: './',
	server: {
		proxy: {
			'/api': {
				target: 'http://localhost:8686',
				changeOrigin: true,
				configure: (proxy: any) => {
					// Backend is expected to be unavailable briefly during docker startup.
					// Swallow proxy error logs to avoid noisy dev output.
					proxy.on('error', () => {})
				},
				rewrite: (path) => path.replace(/^\/api/, '')
			}
		}
	},
	plugins: [
		react(),
		viteStaticCopy({
			targets: [
				{
					src: 'package.json',
					dest: '.'
				},
				{
					src: 'backend/*',
					dest: '.'
				}
			]
		})
	]
})
