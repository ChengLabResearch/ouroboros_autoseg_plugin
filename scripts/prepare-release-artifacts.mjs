import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises'
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

await mkdir(outputRoot, { recursive: true })
const thirdPartyNotices = await frontendThirdPartyNotices()

for (const variant of variants) {
	const artifactRoot = join(outputRoot, variant.directory)
	await rm(artifactRoot, { recursive: true, force: true })
	await cp(dist, artifactRoot, { recursive: true })
	await pruneBuildOnlyFiles(artifactRoot)
	await cp(join(root, 'LICENSE'), join(artifactRoot, 'LICENSE'))
	await cp(join(root, 'backend', 'licenses', 'LICENSE-SAM'), join(artifactRoot, 'LICENSE-SAM'))
	await writeFile(join(artifactRoot, 'THIRD_PARTY_NOTICES.txt'), thirdPartyNotices)
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
		commit: releaseCommit,
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
		'about.hbs',
		'about.toml',
		'app',
		'compose.dev.yml',
		'compose.gpu.dev.yml',
		'compose.gpu.yml',
		'compose.registry.yml',
		'docs',
		'licenses',
		'poetry.lock',
		'pyproject.toml',
		'scripts',
		'src',
		'tests'
	]

	await Promise.all(
		buildOnlyPaths.map((relativePath) =>
			rm(join(artifactRoot, relativePath), { recursive: true, force: true })
		)
	)
}

/**
 * Licence texts for everything bundled into the frontend build: the locked
 * production npm packages and the Poppins web font.
 */
async function frontendThirdPartyNotices() {
	const lockfile = JSON.parse(await readFile(join(root, 'package-lock.json'), 'utf8'))
	const sections = []
	for (const [path, entry] of Object.entries(lockfile.packages)) {
		if (!path || entry.dev || entry.devOptional) continue
		sections.push({
			title: `${path.replace(/^node_modules\//u, '')} ${entry.version} (${entry.license})`,
			text: await packageLicenseText(join(root, path))
		})
	}
	sections.push({
		title: 'Poppins web font (OFL-1.1)',
		text: await readFile(join(root, 'licenses', 'Poppins-OFL.txt'), 'utf8')
	})

	const header = [
		'Third-party software in the Ouroboros automatic segmentation plugin frontend',
		'',
		'The backend container image has its own notices, including the SAM License',
		'for the SAM 3 engine, under /usr/share/licenses/ouroboros-autoseg-plugin-backend/.',
		''
	].join('\n')
	const rule = '='.repeat(80)
	return `${header}${sections
		.map(({ title, text }) => `\n${rule}\n${title}\n${rule}\n\n${text.trim()}\n`)
		.join('')}`
}

async function packageLicenseText(packageRoot) {
	const licenseFile = (await readdir(packageRoot)).find((name) => /^licen[cs]e/iu.test(name))
	if (!licenseFile) throw new Error(`no licence file in ${packageRoot}`)
	return readFile(join(packageRoot, licenseFile), 'utf8')
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
      - PLUGIN_GIT_SHA=${releaseCommit ?? 'unknown'}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    shm_size: 1gb
${gpuReservation}
volumes:
  ouroboros-volume:
    external: true
`
}
