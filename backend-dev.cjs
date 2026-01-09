const { upAll, downAll } = require('docker-compose/dist/v2')
const { join } = require('path')
const { execSync } = require('child_process')

const containerFolder = join(__dirname, 'backend')

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

console.log(`Hardware Check: ${hasGpu ? 'NVIDIA GPU Detected ?' : 'No GPU Detected ?'}`)
console.log(`Starting Docker Container using ${composeFile}...`)

upAll({ 
    cwd: containerFolder, 
    config: composeFile, 
    log: false, 
    commandOptions: ['--build'] 
}).catch((err) => {
    console.error('Failed to start plugin docker environment.', err)
})

async function cleanup() {
    try {
        console.log('Shutting Down Docker Container')
        await downAll({ 
            cwd: containerFolder, 
            config: composeFile,
            log: false 
        })
    } catch (e) {
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
