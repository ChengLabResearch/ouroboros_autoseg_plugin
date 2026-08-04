import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { basename, join, resolve } from 'node:path'

const root = process.cwd()
const archive = resolve(process.argv[2] ?? '')
const expectedVariant = process.argv[3]

assert(process.argv[2], 'usage: npm run validate:archive -- <archive.zip> <cpu|cuda>')
assert(
	expectedVariant === 'cpu' || expectedVariant === 'cuda',
	'archive variant must be cpu or cuda'
)
assert(existsSync(archive), `release archive does not exist: ${archive}`)

runUnzip(['-t', archive])
const entries = new Set(
	runUnzip(['-Z1', archive])
		.split(/\r?\n/u)
		.filter(Boolean)
		.map(normalizeEntry)
)

for (const required of [
	'package.json',
	'plugin-release.json',
	'index.html',
	'icon.svg',
	'compose.yml'
]) {
	assert(entries.has(required), `release archive is missing ${required}`)
}

for (const buildOnly of [
	'Cargo.lock',
	'Cargo.toml',
	'Dockerfile',
	'RUST_SCAFFOLD.md',
	'pyproject.toml',
	'poetry.lock'
]) {
	assert(!entries.has(buildOnly), `release archive includes build-only file ${buildOnly}`)
}
for (const buildOnlyDirectory of ['app/', 'docs/', 'scripts/', 'src/', 'tests/']) {
	assert(
		![...entries].some((entry) => entry.startsWith(buildOnlyDirectory)),
		`release archive includes build-only directory ${buildOnlyDirectory}`
	)
}

const packageJson = readArchivedJson('package.json')
const manifest = readArchivedJson('plugin-release.json')
const compose = readArchivedText('compose.yml')
const indexHtml = readArchivedText('index.html')
const artifactLayout = join(
	root,
	'dist-artifacts',
	`${packageJson.name}-${expectedVariant}`
)

assert(manifest.name === packageJson.name, 'archive package name mismatch')
assert(manifest.pluginName === packageJson.pluginName, 'archive plugin name mismatch')
assert(manifest.packageVersion === packageJson.version, 'archive package version mismatch')
assert(manifest.version === packageJson.version, 'archive manifest version mismatch')
assert(manifest.variant === expectedVariant, 'archive variant mismatch')
assert(manifest.artifactName === basename(archive), 'archive filename mismatch')
assert(manifest.index === packageJson.index, 'archive index path mismatch')
assert(manifest.icon === packageJson.icon, 'archive icon path mismatch')
assert(manifest.dockerCompose === packageJson.dockerCompose, 'archive compose path mismatch')
assert(compose.includes(`image: ${manifest.backendImage}`), 'archive compose image mismatch')

if (expectedVariant === 'cuda') {
	assert(manifest.cuda === true, 'CUDA archive manifest is not marked as CUDA')
	assert(compose.includes('capabilities: [gpu]'), 'CUDA archive is missing GPU reservation')
} else {
	assert(manifest.cuda === false, 'CPU archive manifest is marked as CUDA')
	assert(!compose.includes('capabilities: [gpu]'), 'CPU archive reserves a GPU')
}

const assetPaths = referencedAssets(indexHtml)
assert(assetPaths.length > 0, 'index.html does not reference built assets')
for (const assetPath of assetPaths) {
	assert(entries.has(assetPath), `index.html references missing archive asset ${assetPath}`)
}

for (const file of ['package.json', 'plugin-release.json', 'compose.yml']) {
	const archived = readArchivedText(file).trim()
	const built = (await readFile(join(artifactLayout, file), 'utf8')).trim()
	assert(archived === built, `archived ${file} does not match the validated layout`)
}

console.log(`Validated ${expectedVariant} release archive ${archive}.`)

function readArchivedJson(path) {
	return JSON.parse(readArchivedText(path))
}

function readArchivedText(path) {
	return runUnzip(['-p', archive, path])
}

function referencedAssets(html) {
	return [...html.matchAll(/(?:src|href)=["']\.?\/?(assets\/[^"']+)["']/gu)].map(
		(match) => normalizeEntry(match[1])
	)
}

function normalizeEntry(entry) {
	return entry.replace(/^\.\//u, '').replace(/^\//u, '').replace(/\/$/u, '')
}

function runUnzip(args) {
	const result = spawnSync('unzip', args, { encoding: 'utf8' })
	if (result.status !== 0) {
		throw new Error(result.stderr || result.stdout || 'unzip failed')
	}
	return result.stdout
}

function assert(condition, message) {
	if (!condition) throw new Error(message)
}
