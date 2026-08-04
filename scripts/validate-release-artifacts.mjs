import { readFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { join } from 'node:path'

const root = process.cwd()
const outputRoot = join(root, 'dist-artifacts')
const packageJson = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
const releaseTag =
	process.env.OUROBOROS_AUTOSEG_PLUGIN_RELEASE_TAG ??
	process.env.GITHUB_REF_NAME ??
	`v${packageJson.version}`
const imageRepository =
	process.env.OUROBOROS_AUTOSEG_BACKEND_IMAGE_REPOSITORY ??
	'ghcr.io/chenglabresearch/ouroboros-autoseg-backend'
const imageTag = process.env.OUROBOROS_AUTOSEG_BACKEND_IMAGE_TAG ?? releaseTag
const cpuDigest = process.env.OUROBOROS_AUTOSEG_BACKEND_CPU_DIGEST ?? null
const cudaDigest = process.env.OUROBOROS_AUTOSEG_BACKEND_CUDA_DIGEST ?? null
const releaseCommit =
	process.env.OUROBOROS_AUTOSEG_PLUGIN_COMMIT ?? process.env.GITHUB_SHA ?? null

const variants = [
	{
		name: 'cpu',
		directory: `${packageJson.name}-cpu`,
		artifactName: artifactNameForVariant('cpu'),
		image: imageReference('', cpuDigest),
		cuda: false
	},
	{
		name: 'cuda',
		directory: `${packageJson.name}-cuda`,
		artifactName: artifactNameForVariant('cuda'),
		image: imageReference('-cuda', cudaDigest),
		cuda: true
	}
]

for (const variant of variants) {
	const artifactRoot = join(outputRoot, variant.directory)
	assert(existsSync(artifactRoot), `missing release layout: ${artifactRoot}`)
	for (const required of [
		'package.json',
		'plugin-release.json',
		'index.html',
		'icon.svg',
		'compose.yml'
	]) {
		assert(existsSync(join(artifactRoot, required)), `${variant.name} layout missing ${required}`)
	}
	for (const buildOnly of [
		'Cargo.lock',
		'Cargo.toml',
		'Dockerfile',
		'RUST_SCAFFOLD.md',
		'app',
		'docs',
		'pyproject.toml',
		'poetry.lock',
		'scripts',
		'src',
		'tests'
	]) {
		assert(
			!existsSync(join(artifactRoot, buildOnly)),
			`${variant.name} layout includes build-only path ${buildOnly}`
		)
	}

	const pluginPackage = await readJson(join(artifactRoot, 'package.json'))
	const manifest = await readJson(join(artifactRoot, 'plugin-release.json'))
	const compose = await readFile(join(artifactRoot, 'compose.yml'), 'utf8')
	const indexHtml = await readFile(join(artifactRoot, 'index.html'), 'utf8')

	assert(pluginPackage.name === packageJson.name, `${variant.name} package name mismatch`)
	assert(pluginPackage.pluginName === packageJson.pluginName, `${variant.name} plugin name mismatch`)
	assert(pluginPackage.version === packageJson.version, `${variant.name} package version mismatch`)
	assert(pluginPackage.index === packageJson.index, `${variant.name} package index mismatch`)
	assert(pluginPackage.icon === packageJson.icon, `${variant.name} package icon mismatch`)
	assert(pluginPackage.dockerCompose === packageJson.dockerCompose, `${variant.name} compose path mismatch`)
	assert(manifest.name === packageJson.name, `${variant.name} manifest name mismatch`)
	assert(manifest.pluginName === packageJson.pluginName, `${variant.name} manifest plugin name mismatch`)
	assert(manifest.version === packageJson.version, `${variant.name} manifest version mismatch`)
	assert(manifest.packageVersion === packageJson.version, `${variant.name} package version mismatch`)
	assert(manifest.releaseTag === releaseTag, `${variant.name} release tag mismatch`)
	assert(manifest.artifactName === variant.artifactName, `${variant.name} artifact name mismatch`)
	assert(manifest.variant === variant.name, `${variant.name} manifest variant mismatch`)
	assert(manifest.backendImage === variant.image, `${variant.name} backend image mismatch`)
	assert(manifest.backendImageRepository === imageRepository, `${variant.name} repository mismatch`)
	assert(manifest.backendImageTag === imageTag, `${variant.name} image tag mismatch`)
	assert(manifest.cuda === variant.cuda, `${variant.name} CUDA flag mismatch`)
	assert(manifest.index === packageJson.index, `${variant.name} manifest index mismatch`)
	assert(manifest.icon === packageJson.icon, `${variant.name} manifest icon mismatch`)
	assert(manifest.dockerCompose === packageJson.dockerCompose, `${variant.name} manifest compose mismatch`)
	assert(compose.includes(`image: ${variant.image}`), `${variant.name} compose image mismatch`)
	assert(
		compose.includes(`PLUGIN_GIT_SHA=${releaseCommit ?? 'unknown'}`),
		`${variant.name} compose plugin revision mismatch`
	)
	assert(manifest.commit === releaseCommit, `${variant.name} manifest commit mismatch`)

	const assetPaths = referencedAssets(indexHtml)
	assert(assetPaths.length > 0, `${variant.name} index does not reference built assets`)
	for (const assetPath of assetPaths) {
		assert(
			existsSync(join(artifactRoot, assetPath)),
			`${variant.name} index references missing asset ${assetPath}`
		)
	}

	if (variant.cuda) {
		assert(compose.includes('capabilities: [gpu]'), 'CUDA compose missing GPU reservation')
	} else {
		assert(!compose.includes('capabilities: [gpu]'), 'CPU compose should not reserve a GPU')
	}
}

console.log(`Validated ${variants.length} release artifact layouts for ${releaseTag}.`)

async function readJson(path) {
	return JSON.parse(await readFile(path, 'utf8'))
}

function imageReference(suffix, digest) {
	if (digest) return `${imageRepository}@${digest}`
	return `${imageRepository}:${imageTag}${suffix}`
}

function artifactNameForVariant(variantName) {
	return `${packageJson.name}-${safeFilePart(releaseTag)}-${variantName}.zip`
}

function safeFilePart(value) {
	return value.replaceAll('/', '-')
}

function referencedAssets(html) {
	return [...html.matchAll(/(?:src|href)=["']\.?\/?(assets\/[^"']+)["']/gu)].map(
		(match) => match[1]
	)
}

function assert(condition, message) {
	if (!condition) throw new Error(message)
}
