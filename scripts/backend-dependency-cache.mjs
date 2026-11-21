import { spawnSync } from 'node:child_process'
import { appendFileSync, closeSync, mkdtempSync, openSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
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
	writeFileSync(secretPath, 'probe\n')
	const args = [
		'buildx', 'build', '--builder', process.env.BUILDER,
		'--platform', record.platform, '--file', 'backend/Dockerfile',
		'--build-arg', `${record.variant.toUpperCase()}_BUILDER_IMAGE=${base.image.split('@')[0]}@${base.digest}`,
		'--build-arg', `BACKEND_CACHE_NAMESPACE=${cache.namespace}`,
		'--target', operation === 'probe' ? cache.dependencyStage : cache.exportStage,
		'--progress=rawjson', '--secret', `id=backend-dependency-probe,src=${secretPath}`,
		'--output', operation === 'probe' ? 'type=cacheonly' : `type=local,dest=${resolve(archiveDirectory)}`
	]
	if (cacheFrom) args.push('--cache-from', cacheFrom)
	args.push('backend')
	console.log(`${operation}: ${record.variant} dependency cache; BuildKit log: ${logPath}`)
	const log = openSync(logPath, 'w')
	let result
	try {
		// Export keeps the probe guard: an evicted snapshot must not compile during writeback.
		result = spawnSync('docker', args, { stdio: ['ignore', log, log] })
	} finally {
		closeSync(log)
		unlinkSync(secretPath)
	}
	if (result.error) throw result.error
	if (result.signal) throw new Error(`Dependency cache ${operation} interrupted by ${result.signal}; see ${logPath}`)
	const raw = readFileSync(logPath, 'utf8')
	if (operation === 'probe' && result.status !== 0 && isDependencyCacheMiss(raw)) {
		output('dependency_miss', 'true')
		console.log('Dependency cache miss: the probe stopped before compilation.')
		return
	}
	if (result.status !== 0) {
		const diagnostic = raw.split('\n').findLast((line) => line.startsWith('ERROR:'))
		throw new Error(`Dependency cache ${operation} failed; see ${logPath}${diagnostic ? `\n${diagnostic}` : ''}`)
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
