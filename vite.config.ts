import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy'
// @ts-ignore - Node builtin types are not configured in this repo's tsconfig.node.
import { execSync } from 'child_process'
// @ts-ignore - Node builtin types are not configured in this repo's tsconfig.node.
import { existsSync, readFileSync } from 'fs'
// @ts-ignore - Node builtin types are not configured in this repo's tsconfig.node.
import { join } from 'path'

function detectComposeFile(): string {
	try {
		execSync('nvidia-smi', { stdio: 'ignore' })
		return 'compose.gpu.yml'
	} catch {
		return 'compose.yml'
	}
}

function parseComposePsJson(raw: string): any[] {
	const trimmed = raw.trim()
	if (!trimmed) return []

	try {
		const parsed = JSON.parse(trimmed)
		return Array.isArray(parsed) ? parsed : [parsed]
	} catch {
		// Older/newer compose versions may output one JSON object per line.
		return trimmed
			.split('\n')
			.filter((line) => line.trim().indexOf('{') === 0)
			.map((line) => JSON.parse(line))
	}
}

function dockerStatusPlugin() {
	const composeFile = detectComposeFile()
	const composePath = `backend/${composeFile}`
	const statusFile = join('.', '.docker-build-status.json')

	function readBuildSessionStatus(): any | null {
		if (!existsSync(statusFile)) return null
		try {
			return JSON.parse(readFileSync(statusFile, 'utf8'))
		} catch {
			return null
		}
	}

	const handler = (_req: any, res: any) => {
		let services: any[] = []
		let dockerError: string | null = null
		const session = readBuildSessionStatus()
		const sessionPhase = typeof session?.phase === 'string' ? session.phase : null

		try {
			const output = execSync(
				`docker compose -f ${composePath} ps --all --format json our_autoseg`,
				{ encoding: 'utf8' }
			)
			services = parseComposePsJson(output)
		} catch (error: any) {
			dockerError = error?.message ?? 'Docker query failed'
		}

		const states = services.map((s) =>
			String(s.State ?? s.state ?? '').toLowerCase()
		)
		const hasServices = services.length > 0
		const isRunning = states.indexOf('running') >= 0
		const hasExited = states.some((s) => s === 'exited' || s === 'dead')
		const sessionInProgress =
			sessionPhase === 'initializing' ||
			sessionPhase === 'building' ||
			sessionPhase === 'starting'
		const sessionError = sessionPhase === 'error'
		const sessionRunning = sessionPhase === 'running'

		const composeStepStatus = sessionError
			? 'error'
			: dockerError
			? 'error'
			: sessionRunning
				? 'completed'
				: sessionInProgress
					? 'in_progress'
			: hasServices
				? 'completed'
				: 'in_progress'
		const containerStepStatus = isRunning
			? 'completed'
			: sessionError || hasExited
				? 'error'
				: 'in_progress'

		const payload = {
			is_ready: isRunning,
			initialization_steps: [
				{ name: 'Docker Compose', status: composeStepStatus },
				{ name: 'Container Startup', status: containerStepStatus }
			],
			start_time: null,
			ready_time: null,
			error: session?.error ?? dockerError,
			session_id: session?.session_id ?? null
		}

		res.setHeader('Content-Type', 'application/json')
		res.end(JSON.stringify(payload))
	}

	return {
		name: 'docker-status-plugin',
		configureServer(server: any) {
			server.middlewares.use('/docker-status', handler)
		},
		configurePreviewServer(server: any) {
			server.middlewares.use('/docker-status', handler)
		}
	}
}

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
		dockerStatusPlugin(),
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
