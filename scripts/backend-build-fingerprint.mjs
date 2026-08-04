import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises'
import { dirname, join, relative, resolve } from 'node:path'

const FINGERPRINT_SCHEMA_VERSION = 1
const VARIANT_STAGES = {
	cpu: ['builder', 'runtime'],
	cuda: ['cuda-builder', 'cuda-runtime']
}

const options = parseArguments(process.argv.slice(2))
const root = resolve(options.root ?? process.cwd())
const backendRoot = join(root, 'backend')
const variant = options.variant

if (!Object.hasOwn(VARIANT_STAGES, variant)) {
	throw new Error(`Unsupported backend variant "${variant}". Expected cpu or cuda.`)
}

const dockerfilePath = join(backendRoot, 'Dockerfile')
const dockerfile = await readFile(dockerfilePath, 'utf8')
const stageImages = dockerfileStageImages(dockerfile)
const digestOverrides = parseDigestOverrides(process.env.BACKEND_BASE_IMAGE_DIGESTS)
const baseImages = []

for (const stage of VARIANT_STAGES[variant]) {
	const image = stageImages.get(stage)
	if (!image) throw new Error(`Dockerfile does not define required stage "${stage}"`)

	baseImages.push({
		stage,
		image,
		digest: digestOverrides[stage] ?? resolveImageDigest(image)
	})
}

const inputPaths = [
	join(backendRoot, '.dockerignore'),
	join(backendRoot, 'Cargo.lock'),
	join(backendRoot, 'Cargo.toml'),
	dockerfilePath,
	join(backendRoot, 'src'),
	join(backendRoot, 'tests'),
	join(root, 'scripts', 'backend-build-fingerprint.mjs')
]
const inputFiles = []

for (const inputPath of inputPaths) {
	inputFiles.push(...(await filesUnder(inputPath)))
}

inputFiles.sort((left, right) => relative(root, left).localeCompare(relative(root, right)))

const inputTreeHash = createHash('sha256')
for (const file of inputFiles) {
	const fileStat = await stat(file)
	const filePath = relative(root, file).replaceAll('\\', '/')
	inputTreeHash.update(`${filePath}\0${(fileStat.mode & 0o777).toString(8)}\0`)
	inputTreeHash.update(await readFile(file))
	inputTreeHash.update('\0')
}

const inputs = {
	schemaVersion: FINGERPRINT_SCHEMA_VERSION,
	variant,
	platform: options.platform ?? 'linux/amd64',
	inputTreeDigest: `sha256:${inputTreeHash.digest('hex')}`,
	baseImages
}
const fingerprint = createHash('sha256').update(stableJson(inputs)).digest('hex')
const record = {
	...inputs,
	fingerprint,
	createdBy: 'scripts/backend-build-fingerprint.mjs'
}

if (options.output) {
	const outputPath = resolve(root, options.output)
	await mkdir(dirname(outputPath), { recursive: true })
	await writeFile(outputPath, `${JSON.stringify(record, null, 2)}\n`)
}

process.stdout.write(`${fingerprint}\n`)

function parseArguments(args) {
	const parsed = {}
	for (let index = 0; index < args.length; index += 1) {
		const argument = args[index]
		if (!argument.startsWith('--')) throw new Error(`Unexpected argument: ${argument}`)
		const key = argument.slice(2)
		const value = args[index + 1]
		if (!value || value.startsWith('--')) throw new Error(`Missing value for --${key}`)
		parsed[key.replaceAll('-', '_')] = value
		index += 1
	}

	return {
		variant: parsed.variant,
		platform: parsed.platform,
		output: parsed.output,
		root: parsed.root
	}
}

function dockerfileStageImages(text) {
	const args = new Map()
	const stages = new Map()

	for (const rawLine of text.split(/\r?\n/u)) {
		const line = rawLine.trim()
		const argMatch = /^ARG\s+([A-Za-z_][A-Za-z0-9_]*)(?:=(\S+))?$/u.exec(line)
		if (argMatch?.[2]) args.set(argMatch[1], argMatch[2])

		const fromMatch = /^FROM\s+(\S+)\s+AS\s+(\S+)$/iu.exec(line)
		if (!fromMatch) continue

		const image = fromMatch[1].replace(/^\$\{([^}]+)\}$/u, (_, name) => {
			const value = args.get(name)
			if (!value) throw new Error(`Dockerfile ARG ${name} has no default image value`)
			return value
		})
		stages.set(fromMatch[2], image)
	}

	return stages
}

function resolveImageDigest(image) {
	if (image.includes('@sha256:')) return image.slice(image.indexOf('@') + 1)

	return execFileSync(
		'docker',
		['buildx', 'imagetools', 'inspect', image, '--format', '{{.Manifest.Digest}}'],
		{ encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] }
	).trim()
}

function parseDigestOverrides(value) {
	if (!value) return {}
	const parsed = JSON.parse(value)
	if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
		throw new Error('BACKEND_BASE_IMAGE_DIGESTS must be a JSON object keyed by Docker stage')
	}
	return parsed
}

async function filesUnder(path) {
	const pathStat = await stat(path)
	if (pathStat.isFile()) return [path]
	if (!pathStat.isDirectory()) return []

	const files = []
	const entries = await readdir(path, { withFileTypes: true })
	entries.sort((left, right) => left.name.localeCompare(right.name))
	for (const entry of entries) {
		files.push(...(await filesUnder(join(path, entry.name))))
	}
	return files
}

function stableJson(value) {
	if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`
	if (value && typeof value === 'object') {
		return `{${Object.keys(value)
			.sort()
			.map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`)
			.join(',')}}`
	}
	return JSON.stringify(value)
}
