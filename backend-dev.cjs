const { upAll, downAll } = require('docker-compose')
const { join } = require('path')
const { execSync } = require('child_process')
const { createHash } = require('crypto')
const { existsSync, readFileSync, writeFileSync } = require('fs')

const containerFolder = join(__dirname, 'backend')
const statusFile = join(__dirname, '.docker-build-status.json')
const serviceName = 'our_autoseg'
const sessionId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
const dependencyFiles = ['Dockerfile', 'pyproject.toml', 'poetry.lock']

function fileHash(path) {
    const hash = createHash('sha256')
    hash.update(readFileSync(path))
    return hash.digest('hex')
}

function computeDependencyHash() {
    const hash = createHash('sha256')
    for (const file of dependencyFiles) {
        const filePath = join(containerFolder, file)
        if (!existsSync(filePath)) {
            hash.update(`${file}:missing`)
            continue
        }
        hash.update(file)
        hash.update(':')
        hash.update(fileHash(filePath))
        hash.update(';')
    }
    return hash.digest('hex')
}

function readPreviousStatus() {
    if (!existsSync(statusFile)) {
        return {}
    }
    try {
        return JSON.parse(readFileSync(statusFile, 'utf8'))
    } catch (_e) {
        return {}
    }
}

const dependencyHash = computeDependencyHash()
const previousStatus = readPreviousStatus()
const shouldBuild = previousStatus.dependency_hash !== dependencyHash

const state = {
    session_id: sessionId,
    service: serviceName,
    compose_file: null,
    dependency_hash: dependencyHash,
    rebuild_required: shouldBuild,
    phase: 'initializing',
    started_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    error: null
}

function writeStatus(patch = {}) {
    Object.assign(state, patch)
    state.updated_at = new Date().toISOString()
    try {
        writeFileSync(statusFile, JSON.stringify(state, null, 2))
    } catch (e) {
        console.error('Failed to write docker status file:', e)
    }
}

function checkGpu() {
    try {
        execSync('nvidia-smi', { stdio: 'ignore' })
        return true
    } catch (e) {
        return false
    }
}

const hasGpu = checkGpu()
const composeFile = hasGpu ? 'compose.gpu.dev.yml' : 'compose.dev.yml'
state.compose_file = composeFile

console.log(`Hardware Check: ${hasGpu ? 'NVIDIA GPU Detected ?' : 'No GPU Detected ?'}`)
console.log(`Starting Docker Container using ${composeFile}...`)
if (shouldBuild) {
    console.log('Dependency change detected. Rebuilding image...')
} else {
    console.log('No dependency changes detected. Reusing existing image...')
}
writeStatus({ phase: shouldBuild ? 'building' : 'starting', error: null })

upAll({ 
    cwd: containerFolder, 
    config: composeFile, 
    log: false, 
    commandOptions: shouldBuild ? ['--build'] : []
}).then(() => {
    writeStatus({ phase: 'running', error: null, ready_at: new Date().toISOString() })
}).catch((err) => {
    writeStatus({ phase: 'error', error: String(err) })
    console.error('Failed to start plugin docker environment.', err)
})

async function cleanup() {
    try {
        console.log('Shutting Down Docker Container')
        writeStatus({ phase: 'stopping' })
        await downAll({ 
            cwd: containerFolder, 
            config: composeFile,
            log: false 
        })
        writeStatus({ phase: 'stopped', stopped_at: new Date().toISOString() })
    } catch (e) {
        writeStatus({ phase: 'error', error: String(e) })
        console.error(e)
    }
    process.exit()
}

process.on('SIGINT', async () => {
    try {
        await cleanup()
    } catch (e) {
        console.error(e)
    }
})

setInterval(() => {}, 1000)
