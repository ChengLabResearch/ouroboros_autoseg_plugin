// Branch-only #103 experiment. Derive each arm from the production Dockerfile.
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { parseDockerfile } from '../../scripts/backend-build-fingerprint.mjs'

const [method, variant, scenario] = process.argv.slice(2)
assert(['registry', 'mounts', 'hybrid'].includes(method), 'Expected registry, mounts or hybrid')
assert(['cpu', 'cuda'].includes(variant), 'Expected cpu or cuda')
assert(['seed', 'source', 'dependency'].includes(scenario), 'Unknown spike scenario')
assert(process.env.RUNNER_TEMP && process.env.GITHUB_OUTPUT && process.env.GITHUB_RUN_ID)
const output = join(process.env.RUNNER_TEMP, 'cache-spike')
await mkdir(output, { recursive: true })

let dockerfile = await readFile('backend/Dockerfile', 'utf8')
const parsed = parseDockerfile(dockerfile)
const builder = variant === 'cpu' ? 'builder' : 'cuda-builder'
const runtime = variant === 'cpu' ? 'runtime' : 'cuda-runtime'
const stage = parsed.stages.find(({ name }) => name === builder)
const dependencyCopy = stage.lines.findIndex((line) => line === 'COPY Cargo.toml Cargo.lock ./')
assert(dependencyCopy > 0, 'Cannot identify the native build environment')
const baseArg = `${variant.toUpperCase()}_BUILDER_IMAGE`
// Includes the pinned builder, Rust/toolchain setup, features and CUDA arch.
// Deliberately excludes Cargo.lock so the dependency probe restores the seed.
const environment = createHash('sha256')
	.update(JSON.stringify(['mount-layout-v1', 'linux/amd64', parsed.globalArgs.get(baseArg), stage.lines.slice(0, dependencyCopy)]))
	.digest('hex').slice(0, 24)
const namespace = `issue103-spike-${process.env.GITHUB_RUN_ID}-${variant}-${environment}`
const cachePath = join(output, 'cache')
await mkdir(cachePath, { recursive: true })
const cargoHome = variant === 'cpu' ? '/usr/local/cargo' : '/root/.cargo'
const targets = { target: '/build/plugin/backend/target', registry: `${cargoHome}/registry`, git: `${cargoHome}/git` }
const cacheMap = Object.fromEntries(Object.entries(targets).map(([name, target]) => [
	join(cachePath, name), { target, id: `${namespace}-${name}`, sharing: 'locked' }
]))
const mounts = Object.values(cacheMap).map(({ target, id, sharing }) =>
	`--mount=type=cache,target=${target},id=${id},sharing=${sharing}`).join(' ')
const binary = 'ouroboros-autoseg-plugin-backend'

if (method === 'hybrid') {
	const start = dependencyCopy + 1
	assert(stage.lines[start].startsWith('RUN '), 'Expected dependency RUN after manifests')
	let end = start
	while (stage.lines[end].endsWith('\\')) end += 1
	const environmentStage = `${builder}-environment`
	const dependencyStage = `${builder}-dependencies`
	const prefix = stage.lines.slice(0, dependencyCopy).join('\n')
		.replace(`FROM ${stage.source} AS ${builder}`, `FROM ${stage.source} AS ${environmentStage}`)
	// A mount hides its underlying image path. Snapshot into a separate stage,
	// then restore canonical paths without retaining that stage in the builder.
	const dependencyRun = stage.lines.slice(start, end + 1).join('\n')
		.replace('RUN ', `RUN ${mounts} `)
		.replaceAll('cargo build --release', 'cargo build --verbose --release')
		+ ` \\\n    && snapshot_started="$(date +%s%N)" \\\n    && mkdir -p /cache-spike-snapshot \\\n    && cp -a --parents ${Object.values(targets).join(' ')} /cache-spike-snapshot/ \\\n    && printf 'hybrid-copy-ns=%s\\n' "$(( $(date +%s%N) - snapshot_started ))" \\\n    && du -sb /cache-spike-snapshot`
	const replacement = [
		prefix,
		`FROM ${environmentStage} AS ${dependencyStage}`,
		stage.lines[dependencyCopy], dependencyRun, '',
		`FROM ${environmentStage} AS ${builder}`,
		stage.lines[dependencyCopy],
		`COPY --from=${dependencyStage} /cache-spike-snapshot/ /`,
		...stage.lines.slice(end + 1)
	].join('\n')
	dockerfile = dockerfile.replace(stage.lines.join('\n'), replacement)
}

if (method !== 'registry') {
	// Keep instrumentation and real-source invalidation identical in both arms.
	for (const current of parseDockerfile(dockerfile).stages.filter(({ name }) =>
		name === builder || (variant === 'cpu' && name === 'test'))) {
		const original = current.lines.join('\n')
		const instrumented = original
			.replaceAll('cargo build --release', 'cargo build --verbose --release')
			.replaceAll('cargo test --release', 'cargo test --verbose --release')
			.replace('touch src/lib.rs src/main.rs', 'touch build.rs src/lib.rs src/main.rs')
		dockerfile = dockerfile.replace(original, instrumented)
	}
}

if (method === 'mounts') {
	for (const current of parseDockerfile(dockerfile).stages.filter(({ name }) => name === builder || (variant === 'cpu' && name === 'test'))) {
		const lines = [...current.lines]
		for (let start = 0; start < lines.length; start += 1) {
			if (!lines[start].startsWith('RUN ')) continue
			let end = start
			while (lines[end].endsWith('\\')) end += 1
			let instruction = lines.slice(start, end + 1).join('\n')
			if (!instruction.includes('cargo ')) continue
			instruction = instruction.replace('RUN ', `RUN ${mounts} `)
			if (instruction.includes('touch build.rs src/lib.rs src/main.rs')) {
				instruction += ` \\\n    && mkdir -p /out && cp target/release/${binary} /out/${binary}`
			}
			lines.splice(start, end - start + 1, instruction)
		}
		dockerfile = dockerfile.replace(current.lines.join('\n'), lines.join('\n'))
	}
	const runtimeStage = parsed.stages.find(({ name }) => name === runtime).lines.join('\n')
	dockerfile = dockerfile.replace(runtimeStage, runtimeStage.replace(
		`/build/plugin/backend/target/release/${binary}`, `/out/${binary}`))
}

if (scenario === 'source') {
	const path = 'backend/src/main.rs'
	const source = await readFile(path, 'utf8')
	const original = 'Rust backend scaffold listening'
	const changed = 'Rust backend cache spike listening'
	assert.equal(source.split(original).length, 2, 'Startup-log source probe no longer applies')
	await writeFile(path, source.replace(original, changed))
	// Prove the produced executable contains the edited source, not a stale dummy.
	const current = parseDockerfile(dockerfile).stages.find(({ name }) => name === runtime)
	assert(current.lines.includes('EXPOSE 8686'))
	dockerfile = dockerfile.replace(current.lines.join('\n'), current.lines.join('\n').replace(
		'EXPOSE 8686', `RUN grep -aF '${changed}' /usr/local/bin/${binary} > /dev/null\n\nEXPOSE 8686`))
}

await writeFile(join(output, 'Dockerfile'), dockerfile)
await writeFile(join(output, 'config.json'), `${JSON.stringify({ method, variant, scenario, environment, cacheMap }, null, 2)}\n`)
const values = {
	dockerfile: join(output, 'Dockerfile'), runtime, cache_path: cachePath,
	cache_key: namespace, cache_map: JSON.stringify(cacheMap),
	restore_mounts: (method === 'mounts' && scenario !== 'seed') || (method === 'hybrid' && scenario === 'dependency'),
	extract_mounts: method === 'mounts' || (method === 'hybrid' && scenario === 'seed'),
	save_mounts: method !== 'registry' && scenario === 'seed',
	registry_cache: `ghcr.io/chenglabresearch/ouroboros-autoseg-backend:spike-103-${process.env.GITHUB_RUN_ID}-${variant}-${method}`
}
await appendFile(process.env.GITHUB_OUTPUT, Object.entries(values).map(([key, value]) => `${key}=${value}\n`).join(''))
