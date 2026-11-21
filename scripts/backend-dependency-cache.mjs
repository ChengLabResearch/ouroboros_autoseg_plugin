import { spawnSync } from 'node:child_process'
import { appendFileSync, closeSync, mkdirSync, mkdtempSync, openSync, readFileSync, rmSync, unlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const MISS_MARKER = 'OUROBOROS_DEPENDENCY_CACHE_MISS'

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
	try {
		main()
	} catch (error) {
		console.error(error.message)
		process.exitCode = 1
	}
}

function main() {
	const [operation, recordPath, archiveDirectory] = process.argv.slice(2)
	if (!['probe', 'export'].includes(operation) || !recordPath || !archiveDirectory) {
		throw new Error('Usage: node scripts/backend-dependency-cache.mjs <probe|export> <record.json> <archive-dir>')
	}
	if (!process.env.BUILDER) throw new Error('BUILDER is required')
	const record = JSON.parse(readFileSync(recordPath, 'utf8'))
	const cache = record.dependencyCache
	const builderStage = record.variant === 'cpu' ? 'builder' : 'cuda-builder'
	const base = record.baseImages.find(({ stage }) => stage === builderStage)
	if (!cache || !base || !['cpu', 'cuda'].includes(record.variant)) {
		throw new Error('Fingerprint record lacks the variant dependency-cache inputs')
	}
	const cacheFrom = operation === 'probe'
		? inspectCache(process.env.CANONICAL_CACHE)
		: process.env.CACHE_FROM ?? ''
	if (operation === 'probe') output('cache_from', cacheFrom)

	const temporary = mkdtempSync(join(process.env.RUNNER_TEMP ?? tmpdir(), 'backend-dependency-cache-'))
	const secretPath = join(temporary, 'probe-secret')
	const logPath = join(temporary, `${operation}.jsonl`)
	const tarPath = join(temporary, 'snapshot.tar')
	writeFileSync(secretPath, 'probe\n')
	const args = [
		'buildx', 'build', '--builder', process.env.BUILDER,
		'--platform', record.platform, '--file', 'backend/Dockerfile',
		'--build-arg', `${record.variant.toUpperCase()}_BUILDER_IMAGE=${base.image.split('@')[0]}@${base.digest}`,
		'--build-arg', `BACKEND_CACHE_NAMESPACE=${cache.namespace}`,
		'--target', operation === 'probe' ? cache.dependencyStage : cache.exportStage,
		`--progress=${operation === 'probe' ? 'rawjson' : 'plain'}`, '--secret', `id=backend-dependency-probe,src=${secretPath}`,
		'--output', operation === 'probe' ? 'type=cacheonly' : `type=tar,dest=${tarPath}`
	]
	if (cacheFrom) args.push('--cache-from', cacheFrom)
	args.push('backend')
	console.log(`${operation}: ${record.variant} dependency cache${operation === 'probe' ? `; BuildKit log: ${logPath}` : ''}`)
	const log = operation === 'probe' ? openSync(logPath, 'w') : undefined
	let result
	try {
		// Export keeps the probe guard: an evicted snapshot must not compile during writeback.
		// Large local-directory exports stalled in CI and a real CPU reproduction.
		// Native tar export avoids fsutil's per-file receive path; bound writeback
		// so a stalled transfer cannot hold the verified candidate indefinitely.
		result = spawnSync('docker', args, {
			stdio: log === undefined ? 'inherit' : ['ignore', log, log],
			...(operation === 'export' ? { timeout: 300_000, killSignal: 'SIGKILL' } : {})
		})
	} finally {
		if (log !== undefined) closeSync(log)
		unlinkSync(secretPath)
	}
	if (operation === 'export' && result.status !== 0) rmSync(tarPath, { force: true })
	if (result.error?.code === 'ETIMEDOUT') throw new Error('Dependency snapshot export exceeded five minutes')
	if (result.error) throw result.error
	if (result.signal) throw new Error(`Dependency cache ${operation} interrupted by ${result.signal}`)
	const raw = operation === 'probe' ? readFileSync(logPath, 'utf8') : ''
	if (operation === 'probe' && result.status !== 0 && isDependencyCacheMiss(raw)) {
		output('dependency_miss', 'true')
		console.log('Dependency cache miss: the probe stopped before compilation.')
		return
	}
	if (result.status !== 0) {
		const diagnostic = raw.split('\n').findLast((line) => line.startsWith('ERROR:'))
		throw new Error(`Dependency cache ${operation} failed; see ${operation === 'probe' ? logPath : 'BuildKit output above'}${diagnostic ? `\n${diagnostic}` : ''}`)
	}
	if (operation === 'export') {
		mkdirSync(archiveDirectory, { recursive: true })
		try {
			const extracted = spawnSync('tar', ['-xf', tarPath, '--no-same-owner', '-C', resolve(archiveDirectory)], { stdio: 'inherit' })
			if (extracted.error) throw extracted.error
			if (extracted.status !== 0) throw new Error('Dependency snapshot extraction failed')
		} finally {
			unlinkSync(tarPath)
		}
	}
	if (operation === 'probe') output('dependency_miss', 'false')
	console.log(`Dependency cache ${operation} succeeded.`)
}

function inspectCache(reference) {
	if (!reference) throw new Error('CANONICAL_CACHE is required for probe')
	const result = spawnSync('docker', ['buildx', 'imagetools', 'inspect', reference, '--format', '{{.Manifest.Digest}}'], { encoding: 'utf8' })
	if (result.error) throw result.error
	if (result.status !== 0) {
		if (/\b(?:not found|manifest unknown)\b/iu.test(result.stderr)) return ''
		throw new Error(`Unable to inspect dependency cache ${reference}: ${result.stderr.trim()}`)
	}
	const digest = result.stdout.trim()
	if (!/^sha256:[a-f0-9]{64}$/u.test(digest)) throw new Error(`Invalid registry cache digest: ${digest}`)
	const repository = reference.split('@')[0].replace(/:[^/]+$/u, '')
	return `type=registry,ref=${repository}@${digest}`
}

export function isDependencyCacheMiss(raw) {
	const logs = new Map()
	const errors = new Map()
	for (const line of raw.split('\n')) {
		let event
		try {
			event = JSON.parse(line)
		} catch {
			continue // Buildx appends ordinary diagnostics after the raw JSON events.
		}
		for (const log of event.logs ?? []) {
			logs.set(log.vertex, (logs.get(log.vertex) ?? '') + Buffer.from(log.data, 'base64').toString('utf8'))
		}
		for (const vertex of event.vertexes ?? []) {
			if (vertex.error) errors.set(vertex.digest, vertex.error)
		}
	}
	// RUN descriptions contain the marker too; only actual output from the failed vertex counts.
	return errors.size > 0 && [...errors].every(([vertex, error]) =>
		/exit code: 83$/u.test(error) && (logs.get(vertex) ?? '').split(/\r?\n/u).includes(MISS_MARKER))
}

function output(name, value) {
	const line = `${name}=${value}\n`
	if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, line)
	process.stdout.write(line)
}
