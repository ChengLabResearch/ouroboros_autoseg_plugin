// Diagnostic branch only. The production Dockerfile and cache action remain authoritative.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const [variant, scenario] = process.argv.slice(2)
assert(['cpu', 'cuda'].includes(variant), 'Expected cpu or cuda')
assert(['seed', 'source', 'dependency', 'dockerfile'].includes(scenario), 'Unknown measurement scenario')
assert(process.env.RUNNER_TEMP && process.env.GITHUB_OUTPUT && process.env.GITHUB_ENV)
const productionRef = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim()
assert.equal(productionRef, process.env.PRODUCTION_REF, 'Use the exact production commit SHA')
const measurements = join(process.env.RUNNER_TEMP, 'measurements')
const records = join(process.env.RUNNER_TEMP, 'backend-fingerprints')
await mkdir(measurements, { recursive: true })
await mkdir(records, { recursive: true })

async function fingerprint(path) {
	execFileSync(process.execPath, ['scripts/backend-build-fingerprint.mjs', '--variant', variant, '--output', path], { stdio: 'inherit' })
	return JSON.parse(await readFile(path, 'utf8'))
}

const before = await fingerprint(join(measurements, 'baseline-record.json'))
const { parseDockerfile } = await import(pathToFileURL(resolve('scripts/backend-build-fingerprint.mjs')))
if (scenario === 'source') {
	const path = 'backend/src/main.rs'
	const source = await readFile(path, 'utf8')
	assert.equal(source.split('Rust backend scaffold listening').length, 2, 'Startup-log source probe no longer applies')
	await writeFile(path, source.replace('Rust backend scaffold listening', 'Rust backend cache measurement listening'))
	// Runtime-only validation: reject a stale executable without invalidating dependencies.
	const dockerfile = await readFile('backend/Dockerfile', 'utf8')
	const runtime = parseDockerfile(dockerfile).stages.find(({ name }) => name === (variant === 'cpu' ? 'runtime' : 'cuda-runtime'))
	assert(runtime, 'Expected the production runtime stage')
	const original = runtime.lines.join('\n')
	assert.equal(original.split('EXPOSE 8686').length, 2)
	const changed = original.replace('EXPOSE 8686', "RUN grep -aF 'Rust backend cache measurement listening' /usr/local/bin/ouroboros-autoseg-plugin-backend > /dev/null\n\nEXPOSE 8686")
	await writeFile('backend/Dockerfile', dockerfile.replace(original, changed))
} else if (scenario === 'dependency') {
	execFileSync('git', ['apply', fileURLToPath(new URL('./pin-project-lite.patch', import.meta.url))], { stdio: 'inherit' })
} else if (scenario === 'dockerfile') {
	const path = 'backend/Dockerfile'
	const source = await readFile(path, 'utf8')
	const stage = parseDockerfile(source).stages.find(({ name }) => name === (variant === 'cpu' ? 'builder-tools' : 'cuda-builder-tools'))
	assert(stage, 'Expected the production tools stage')
	const original = stage.lines.join('\n')
	const command = 'cargo install cargo-about --version "${CARGO_ABOUT_VERSION}" --locked --features cli'
	assert.equal(original.split(command).length, 2, 'Cargo-about command no longer matches')
	const changed = original.replace(command, 'cargo install cargo-about --locked --version "${CARGO_ABOUT_VERSION}" --features cli')
	await writeFile(path, source.replace(original, changed))
}

const recordPath = join(records, `${variant}.json`)
const record = await fingerprint(recordPath)
assert.equal(record.dependencyCache.namespace, before.dependencyCache.namespace, 'The probe must preserve native cache compatibility')
if (scenario === 'source') assert.deepEqual(record.dependencyCache, before.dependencyCache)
if (['dependency', 'dockerfile'].includes(scenario)) assert.notEqual(record.dependencyCache.key, before.dependencyCache.key)
if (scenario !== 'seed') assert.notEqual(record.fingerprint, before.fingerprint)

// The production cold PR tests real artifact delivery. These jobs recompute
// records after their local mutation, so disable only that download step.
const actionPath = '.github/actions/backend-dependency-cache/action.yml'
const action = await readFile(actionPath, 'utf8')
const download = "    - name: Download canonical build records\n      if: inputs.phase == 'restore'\n"
assert.equal(action.split(download).length, 2, 'Canonical record-download step no longer matches')
await writeFile(actionPath, action.replace(download, "    - name: Download canonical build records\n      if: false # Diagnostic branch: records were generated locally.\n"))

if (scenario !== 'seed') {
	const seed = JSON.parse(await readFile(join(process.env.RUNNER_TEMP, 'seed-metadata', 'seed.json'), 'utf8'))
	assert.equal(seed.productionRef, productionRef)
	assert.equal(seed.variant, variant)
	assert.equal(seed.namespace, record.dependencyCache.namespace)
	assert.match(seed.cacheFrom, /^type=registry,ref=ghcr[.]io\/chenglabresearch\/ouroboros-autoseg-backend@sha256:[a-f0-9]{64}$/)
	await appendFile(process.env.GITHUB_ENV, `CANONICAL_CACHE=${seed.cacheFrom.slice('type=registry,ref='.length)}\n`)
}

await writeFile(join(measurements, 'record.json'), `${JSON.stringify(record, null, 2)}\n`)
await writeFile(join(measurements, 'Dockerfile'), await readFile('backend/Dockerfile', 'utf8'))
await writeFile(join(measurements, 'input-change.patch'), execFileSync('git', ['diff', '--', 'backend/Cargo.lock', 'backend/src/main.rs', 'backend/Dockerfile', actionPath], { encoding: 'utf8' }))
await writeFile(join(measurements, 'scenario.json'), `${JSON.stringify({ productionRef, variant, scenario, diagnosticAdaptation: 'Only the composite record-download step is disabled; all cache behavior is production code.' }, null, 2)}\n`)
const outputs = { record: recordPath, key: record.dependencyCache.key, namespace: record.dependencyCache.namespace, platform: record.platform, runtime: variant === 'cpu' ? 'runtime' : 'cuda-runtime' }
await appendFile(process.env.GITHUB_OUTPUT, Object.entries(outputs).map(([key, value]) => `${key}=${value}\n`).join(''))
