import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { buildFingerprint } from './backend-build-fingerprint.mjs'

const BASE_DIGESTS = {
	builder: `sha256:${'1'.repeat(64)}`,
	runtime: `sha256:${'2'.repeat(64)}`,
	'cuda-builder': `sha256:${'3'.repeat(64)}`,
	'cuda-runtime': `sha256:${'4'.repeat(64)}`
}

const DOCKERFILE = `ARG CPU_BUILDER_IMAGE=example/cpu-builder:1
ARG CPU_RUNTIME_IMAGE=example/cpu-runtime:1
ARG CUDA_BUILDER_IMAGE=example/cuda-builder:1
ARG CUDA_RUNTIME_IMAGE=example/cuda-runtime:1

FROM \${CPU_BUILDER_IMAGE} AS builder
ARG CANDLE_SAM3_COMMIT=shared-pin
RUN cpu-build

FROM builder AS test
RUN cpu-test

# CUDA builder stage
FROM \${CUDA_BUILDER_IMAGE} AS cuda-builder
ARG CANDLE_SAM3_COMMIT=shared-pin
ARG CANDLE_FEATURES=cuda
RUN cuda-build

FROM \${CPU_RUNTIME_IMAGE} AS runtime
COPY --from=builder /backend /backend

FROM \${CUDA_RUNTIME_IMAGE} AS cuda-runtime
COPY --from=cuda-builder /backend /backend
`

describe('backend build fingerprints', () => {
	let root

	beforeEach(async () => {
		root = await mkdtemp(join(tmpdir(), 'backend-fingerprint-'))
		await mkdir(join(root, 'backend', 'src'), { recursive: true })
		await mkdir(join(root, 'backend', 'tests'), { recursive: true })
		await mkdir(join(root, 'scripts'), { recursive: true })
		await Promise.all([
			writeFile(join(root, 'backend', '.dockerignore'), 'target\n'),
			writeFile(join(root, 'backend', 'Cargo.lock'), 'lock-v1\n'),
			writeFile(join(root, 'backend', 'Cargo.toml'), 'manifest-v1\n'),
			writeFile(join(root, 'backend', 'Dockerfile'), DOCKERFILE),
			writeFile(join(root, 'backend', 'src', 'lib.rs'), 'source-v1\n'),
			writeFile(join(root, 'backend', 'tests', 'smoke.rs'), 'test-v1\n'),
			writeFile(join(root, 'scripts', 'backend-build-fingerprint.mjs'), 'fingerprint-code-v1\n')
		])
	})

	afterEach(async () => {
		await rm(root, { recursive: true, force: true })
	})

	const fingerprint = async (variant, options = {}) =>
		buildFingerprint({
			root,
			variant,
			digestOverrides: options.digestOverrides ?? BASE_DIGESTS,
			resolveDigest: () => {
				throw new Error('fixture base-image digest was not overridden')
			}
		})

	it('is deterministic and records each target stage closure', async () => {
		const firstCpu = await fingerprint('cpu')
		const secondCpu = await fingerprint('cpu')
		const cuda = await fingerprint('cuda')

		expect(secondCpu.fingerprint).toBe(firstCpu.fingerprint)
		expect(firstCpu.fingerprint).not.toBe(cuda.fingerprint)
		expect(firstCpu.dockerfile.stages).toEqual(['builder', 'test', 'runtime'])
		expect(cuda.dockerfile.stages).toEqual(['cuda-builder', 'cuda-runtime'])
	})

	it('changes only CUDA when CUDA features or bases change', async () => {
		const cpuBefore = await fingerprint('cpu')
		const cudaBefore = await fingerprint('cuda')
		await writeFile(
			join(root, 'backend', 'Dockerfile'),
			DOCKERFILE.replace('CANDLE_FEATURES=cuda', 'CANDLE_FEATURES=cuda,cudnn').replace(
				'example/cuda-runtime:1',
				'example/cuda-runtime:2'
			).replace('# CUDA builder stage', '# CUDA/cuDNN builder stage')
		)

		const cpuAfter = await fingerprint('cpu')
		const cudaAfter = await fingerprint('cuda')
		expect(cpuAfter.fingerprint).toBe(cpuBefore.fingerprint)
		expect(cudaAfter.fingerprint).not.toBe(cudaBefore.fingerprint)
	})

	it.each([
		['CPU builder', 'RUN cpu-build', 'RUN cpu-build-v2'],
		['CPU test gate', 'RUN cpu-test', 'RUN cpu-test-v2']
	])('changes only CPU for a %s edit', async (_label, before, after) => {
		const cpuBefore = await fingerprint('cpu')
		const cudaBefore = await fingerprint('cuda')
		await writeFile(join(root, 'backend', 'Dockerfile'), DOCKERFILE.replace(before, after))

		const cpuAfter = await fingerprint('cpu')
		const cudaAfter = await fingerprint('cuda')
		expect(cpuAfter.fingerprint).not.toBe(cpuBefore.fingerprint)
		expect(cudaAfter.fingerprint).toBe(cudaBefore.fingerprint)
	})

	it('changes both variants for shared source inputs', async () => {
		const cpuBefore = await fingerprint('cpu')
		const cudaBefore = await fingerprint('cuda')
		await writeFile(join(root, 'backend', 'src', 'lib.rs'), 'source-v2\n')

		const cpuAfter = await fingerprint('cpu')
		const cudaAfter = await fingerprint('cuda')
		expect(cpuAfter.fingerprint).not.toBe(cpuBefore.fingerprint)
		expect(cudaAfter.fingerprint).not.toBe(cudaBefore.fingerprint)
	})

	it('scopes resolved base-image digest changes by variant', async () => {
		const cpuBefore = await fingerprint('cpu')
		const cudaBefore = await fingerprint('cuda')
		const changedDigests = {
			...BASE_DIGESTS,
			'cuda-builder': `sha256:${'5'.repeat(64)}`
		}

		const cpuAfter = await fingerprint('cpu', { digestOverrides: changedDigests })
		const cudaAfter = await fingerprint('cuda', { digestOverrides: changedDigests })
		expect(cpuAfter.fingerprint).toBe(cpuBefore.fingerprint)
		expect(cudaAfter.fingerprint).not.toBe(cudaBefore.fingerprint)
	})
})
