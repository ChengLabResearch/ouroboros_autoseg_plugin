import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

const root = process.cwd()
const dist = join(root, 'dist')
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

await mkdir(outputRoot, { recursive: true })

for (const variant of variants) {
	const artifactRoot = join(outputRoot, variant.directory)
	await rm(artifactRoot, { recursive: true, force: true })
	await cp(dist, artifactRoot, { recursive: true })
	await pruneBuildOnlyFiles(artifactRoot)
	await writeFile(join(artifactRoot, 'compose.yml'), composeForVariant(variant))
	await writeFile(
		join(artifactRoot, 'plugin-release.json'),
		`${JSON.stringify(manifestForVariant(variant), null, 2)}\n`
	)
}

function imageReference(suffix, digest) {
	if (digest) return `${imageRepository}@${digest}`
	return `${imageRepository}:${imageTag}${suffix}`
}

function manifestForVariant(variant) {
	return {
		name: packageJson.name,
		pluginName: packageJson.pluginName,
		version: packageJson.version,
		packageVersion: packageJson.version,
		releaseTag,
		artifactName: variant.artifactName,
		variant: variant.name,
		index: packageJson.index,
		icon: packageJson.icon,
		dockerCompose: packageJson.dockerCompose,
		backendImage: variant.image,
		backendImageRepository: imageRepository,
		backendImageTag: imageTag,
		cuda: variant.cuda,
		commit: process.env.GITHUB_SHA ?? null,
		ref: process.env.GITHUB_REF_NAME ?? null
	}
}

function artifactNameForVariant(variantName) {
	return `${packageJson.name}-${safeFilePart(releaseTag)}-${variantName}.zip`
}

function safeFilePart(value) {
	return value.replaceAll('/', '-')
}

async function pruneBuildOnlyFiles(artifactRoot) {
	const buildOnlyPaths = [
		'.dockerignore',
		'Cargo.lock',
		'Cargo.toml',
		'Dockerfile',
		'RUST_SCAFFOLD.md',
		'app',
		'compose.dev.yml',
		'compose.gpu.dev.yml',
		'compose.gpu.yml',
		'compose.registry.yml',
		'docs',
		'poetry.lock',
		'pyproject.toml',
		'src',
		'tests'
	]

	await Promise.all(
		buildOnlyPaths.map((relativePath) =>
			rm(join(artifactRoot, relativePath), { recursive: true, force: true })
		)
	)
}

function composeForVariant(variant) {
	const gpuReservation = variant.cuda
		? `    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
`
		: ''

	return `services:
  our_autoseg:
    image: ${variant.image}
    ports:
      - "8686:8686"
    volumes:
      - ouroboros-volume:/ouroboros-volume
    environment:
      - VOLUME_MOUNT_PATH=/ouroboros-volume
      - VOLUME_SERVER_URL=http://host.docker.internal:3001
    extra_hosts:
      - "host.docker.internal:host-gateway"
    shm_size: 1gb
${gpuReservation}
volumes:
  ouroboros-volume:
    external: true
`
}
