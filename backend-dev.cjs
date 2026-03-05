const { upAll, downAll } = require('docker-compose')
const { join } = require('path')
const { execSync } = require('child_process')
const { writeFileSync } = require('fs')

const containerFolder = join(__dirname, 'backend')
const statusFile = join(__dirname, '.docker-build-status.json')
const serviceName = 'our_autoseg'
const sessionId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
const state = {
    session_id: sessionId,
    service: serviceName,
    compose_file: null,
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
const composeFile = hasGpu ? 'compose.gpu.yml' : 'compose.yml'
state.compose_file = composeFile

console.log(`Hardware Check: ${hasGpu ? 'NVIDIA GPU Detected ?' : 'No GPU Detected ?'}`)
console.log(`Starting Docker Container using ${composeFile}...`)
writeStatus({ phase: 'building', error: null })

upAll({ 
    cwd: containerFolder, 
    config: composeFile, 
    log: false, 
    commandOptions: ['--build'] 
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
