import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises'
import { dirname, join, relative, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const FINGERPRINT_SCHEMA_VERSION = 2
const VARIANTS = {
	cpu: {
		nativeStage: 'builder-base',
		dependencyStage: 'builder-dependency-layer',
		exportStage: 'cpu-cache-export',
		cargoHome: '/usr/local/cargo',
		baseStages: ['builder', 'runtime'],
		targetStages: ['test', 'runtime'],
		inputPaths: ['tests', 'licenses/LICENSE', 'licenses/LICENSE-SAM']
	},
	cuda: {
		nativeStage: 'cuda-builder-base',
		dependencyStage: 'cuda-builder-dependency-layer',
		exportStage: 'cuda-cache-export',
		cargoHome: '/root/.cargo',
		baseStages: ['cuda-builder', 'cuda-runtime'],
		targetStages: ['cuda-runtime'],
		inputPaths: ['licenses']
	}
}

const invokedPath = process.argv[1]
if (invokedPath && import.meta.url === pathToFileURL(invokedPath).href) {
	await main()
}

async function main() {
	const options = parseArguments(process.argv.slice(2))
	const root = resolve(options.root ?? process.cwd())
	const record = await buildFingerprint({
		root,
		variant: options.variant,
		platform: options.platform,
		digestOverrides: parseDigestOverrides(process.env.BACKEND_BASE_IMAGE_DIGESTS)
	})

	if (options.output) {
		const outputPath = resolve(root, options.output)
		await mkdir(dirname(outputPath), { recursive: true })
		await writeFile(outputPath, `${JSON.stringify(record, null, 2)}\n`)
	}

	process.stdout.write(`${record.fingerprint}\n`)
}

export async function buildFingerprint({
	root: rootOption = process.cwd(),
	variant,
	platform = 'linux/amd64',
	digestOverrides = {},
	resolveDigest = resolveImageDigest
}) {
	const variantConfig = VARIANTS[variant]
	if (!variantConfig) {
		throw new Error(`Unsupported backend variant "${variant}". Expected cpu or cuda.`)
	}

	const root = resolve(rootOption)
	const backendRoot = join(root, 'backend')
	const dockerfilePath = join(backendRoot, 'Dockerfile')
	const dockerfile = await readFile(dockerfilePath, 'utf8')
	const parsedDockerfile = parseDockerfile(dockerfile)
	const stageImages = dockerfileStageImages(parsedDockerfile)
	const baseImages = []

	for (const stage of variantConfig.baseStages) {
		const image = stageImages.get(stage)
		if (!image) throw new Error(`Dockerfile does not define required stage "${stage}"`)

		baseImages.push({
			stage,
			image,
			digest: digestOverrides[stage] ?? resolveDigest(image)
		})
	}

	const dockerfileInput = dockerfileTargetInput(parsedDockerfile, variantConfig.targetStages)
	const dockerfileDigest = `sha256:${createHash('sha256')
		.update(stableJson(dockerfileInput))
		.digest('hex')}`
	const inputPaths = [
		join(backendRoot, '.dockerignore'),
		join(backendRoot, 'Cargo.lock'),
		join(backendRoot, 'Cargo.toml'),
		join(backendRoot, 'build.rs'),
		join(backendRoot, 'about.hbs'),
		join(backendRoot, 'about.toml'),
		join(backendRoot, 'src'),
		...variantConfig.inputPaths.map((path) => join(backendRoot, path)),
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
		platform,
		inputTreeDigest: `sha256:${inputTreeHash.digest('hex')}`,
		dockerfile: {
			digest: dockerfileDigest,
			stages: dockerfileInput.stages.map(({ name }) => name)
		},
		baseImages
	}
	const fingerprint = createHash('sha256').update(stableJson(inputs)).digest('hex')

	return {
		...inputs,
		fingerprint,
		dependencyCache: await dependencyCacheIdentity({
			backendRoot, parsedDockerfile, variant, variantConfig, platform, builderImage: baseImages[0]
		}),
		createdBy: 'scripts/backend-build-fingerprint.mjs'
	}
}

// This is cache compatibility, not a claim that BuildKit has a cached layer.
// Keep the native namespace stable across lockfile and cargo-about changes;
// immutable archives use a dependency generation with a native restore prefix.
async function dependencyCacheIdentity({
	backendRoot, parsedDockerfile, variant, variantConfig, platform, builderImage
}) {
	const hash = (value) => createHash('sha256').update(stableJson(value)).digest('hex')
	const native = hash({
		platform,
		variant,
		builderImage,
		dockerfile: dockerfileTargetInput(parsedDockerfile, [variantConfig.nativeStage])
	})
	const manifestInputs = {}
	for (const path of ['.dockerignore', 'Cargo.toml', 'Cargo.lock']) {
		manifestInputs[path] = await readFile(join(backendRoot, path), 'utf8')
	}
	const generation = hash({
		native,
		manifestInputs,
		dockerfile: dockerfileTargetInput(parsedDockerfile, [variantConfig.dependencyStage])
	})
	const namespace = `backend-cargo-v1-${variant}-${native}`
	return {
		namespace,
		key: `${namespace}-${generation}`,
		restorePrefix: `${namespace}-`,
		dependencyStage: variantConfig.dependencyStage,
		exportStage: variantConfig.exportStage,
		cargoHome: variantConfig.cargoHome
	}
}

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

export function parseDockerfile(text) {
	const directives = []
	const globalArgs = new Map()
	const stages = []
	let currentStage = null
	let sawStage = false

	for (const rawLine of text.split(/\r?\n/u)) {
		const line = rawLine.trim()
		if (!sawStage) {
			const directiveMatch = /^#\s*(syntax|escape|check)\s*=/iu.exec(line)
			if (directiveMatch) directives.push(line)

			const argMatch = /^ARG\s+([A-Za-z_][A-Za-z0-9_]*)(?:=(\S+))?$/u.exec(line)
			if (argMatch?.[2]) globalArgs.set(argMatch[1], argMatch[2])
		}

		const fromMatch = /^FROM(?:\s+--platform=\S+)?\s+(\S+)\s+AS\s+(\S+)$/iu.exec(line)
		if (fromMatch) {
			const stagePreamble = []
			while (currentStage) {
				const priorLine = currentStage.lines.at(-1)?.trim()
				if (priorLine && !priorLine.startsWith('#')) break
				stagePreamble.unshift(currentStage.lines.pop())
			}
			sawStage = true
			currentStage = {
				name: fromMatch[2],
				source: fromMatch[1],
				lines: [...stagePreamble, rawLine]
			}
			stages.push(currentStage)
			continue
		}

		if (currentStage) currentStage.lines.push(rawLine)
	}

	return { directives, globalArgs, stages }
}

function dockerfileStageImages(parsedDockerfile) {
	const stages = new Map()
	for (const stage of parsedDockerfile.stages) {
		const sourceStage = localStage(parsedDockerfile, stage.source)
		const image = sourceStage ? stages.get(sourceStage) : stage.source.replace(/\$\{([^}]+)\}/gu, (_, name) => {
			const value = parsedDockerfile.globalArgs.get(name)
			if (!value) throw new Error(`Dockerfile ARG ${name} has no default image value`)
			return value
		})
		stages.set(stage.name, image)
	}
	return stages
}

function localStage(parsedDockerfile, reference) {
	if (parsedDockerfile.stages.some(({ name }) => name === reference)) return reference
	if (!/^\d+$/u.test(reference)) return null
	return parsedDockerfile.stages[Number.parseInt(reference, 10)]?.name ?? null
}

export function dockerfileTargetInput(parsedDockerfile, targetStages) {
	const stagesByName = new Map(parsedDockerfile.stages.map((stage) => [stage.name, stage]))
	const selectedStages = new Set()

	const visit = (stageName) => {
		if (selectedStages.has(stageName)) return
		const stage = stagesByName.get(stageName)
		if (!stage) throw new Error(`Dockerfile does not define target stage "${stageName}"`)

		const sourceStage = localStage(parsedDockerfile, stage.source)
		if (sourceStage) visit(sourceStage)

		const definition = stage.lines.join('\n')
		for (const match of definition.matchAll(/--from=(?:"([^"]+)"|'([^']+)'|([^\s\\]+))/gu)) {
			const referencedStage = localStage(parsedDockerfile, match[1] ?? match[2] ?? match[3])
			if (referencedStage) visit(referencedStage)
		}

		selectedStages.add(stageName)
	}

	for (const targetStage of targetStages) visit(targetStage)

	const stages = parsedDockerfile.stages
		.filter(({ name }) => selectedStages.has(name))
		.map((stage) => ({ name: stage.name, definition: stage.lines.join('\n') }))
	const referencedGlobalArgs = new Set()
	for (const stage of stages) {
		for (const match of stage.definition.matchAll(/\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)/gu)) {
			const name = match[1] ?? match[2]
			if (parsedDockerfile.globalArgs.has(name)) referencedGlobalArgs.add(name)
		}
	}

	return {
		directives: parsedDockerfile.directives,
		globalArgs: Object.fromEntries(
			[...parsedDockerfile.globalArgs].filter(([name]) => referencedGlobalArgs.has(name))
		),
		stages
	}
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
