import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy'
// @ts-expect-error - Node builtin types are not configured in this repo's tsconfig.node.
import { execSync } from 'child_process'
// @ts-expect-error - Node builtin types are not configured in this repo's tsconfig.node.
import { existsSync, readFileSync } from 'fs'
// @ts-expect-error - Node builtin types are not configured in this repo's tsconfig.node.
import { join } from 'path'

type ComposeService = {
	State?: string
	state?: string
}

type BuildSessionStatus = {
	phase?: string
	error?: string | null
	session_id?: string | null
}

type MiddlewareResponse = {
	setHeader: (name: string, value: string) => void
	end: (body: string) => void
}

type MiddlewareHandler = (_req: unknown, res: MiddlewareResponse) => void

type ViteServerLike = {
	middlewares: {
		use: (path: string, handler: MiddlewareHandler) => void
	}
}

function asComposeService(value: unknown): ComposeService | null {
	if (!value || typeof value !== 'object') return null
	const raw = value as Record<string, unknown>
	const state = typeof raw.state === 'string' ? raw.state : undefined
	const State = typeof raw.State === 'string' ? raw.State : undefined
	return { state, State }
}

function detectComposeFile(): string {
	try {
		execSync('nvidia-smi', { stdio: 'ignore' })
		return 'compose.gpu.yml'
	} catch {
		return 'compose.yml'
	}
}

function parseComposePsJson(raw: string): ComposeService[] {
	const trimmed = raw.trim()
	if (!trimmed) return []

	try {
		const parsed: unknown = JSON.parse(trimmed)
		if (Array.isArray(parsed)) {
			return parsed
				.map(asComposeService)
				.filter((service): service is ComposeService => Boolean(service))
		}
		const service = asComposeService(parsed)
		return service ? [service] : []
	} catch {
		// Older/newer compose versions may output one JSON object per line.
		return trimmed
			.split('\n')
			.filter((line) => line.trim().indexOf('{') === 0)
			.map((line) => {
				try {
					return asComposeService(JSON.parse(line))
				} catch {
					return null
				}
			})
			.filter((service): service is ComposeService => Boolean(service))
	}
}

function dockerStatusPlugin() {
	const composeFile = detectComposeFile()
	const composePath = `backend/${composeFile}`
	const statusFile = join('.', '.docker-build-status.json')

	function readBuildSessionStatus(): BuildSessionStatus | null {
		if (!existsSync(statusFile)) return null
		try {
			const parsed: unknown = JSON.parse(readFileSync(statusFile, 'utf8'))
			if (!parsed || typeof parsed !== 'object') return null
			return parsed as BuildSessionStatus
		} catch {
			return null
		}
	}

	const handler: MiddlewareHandler = (_req, res) => {
		let services: ComposeService[] = []
		let dockerError: string | null = null
		const session = readBuildSessionStatus()
		const sessionPhase = typeof session?.phase === 'string' ? session.phase : null

		try {
			const output = execSync(
				`docker compose -f ${composePath} ps --all --format json our_autoseg`,
				{ encoding: 'utf8' }
			)
			services = parseComposePsJson(output)
		} catch (error: unknown) {
			if (error instanceof Error) {
				dockerError = error.message
			} else {
				dockerError = 'Docker query failed'
			}
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
		configureServer(server: ViteServerLike) {
			server.middlewares.use('/docker-status', handler)
		},
		configurePreviewServer(server: ViteServerLike) {
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
				configure: (proxy) => {
					// Backend is expected to be unavailable briefly during docker startup.
					// Swallow proxy error logs to avoid noisy dev output.
					const proxyWithOn = proxy as { on?: (event: string, listener: () => void) => void }
					proxyWithOn.on?.('error', () => {
						return
					})
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
