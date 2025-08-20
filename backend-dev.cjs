const { upAll, downAll } = require('docker-compose/dist/v2')
const { join } = require('path')

const containerFolder = join(__dirname, 'backend')

// Start docker compose
console.log('Starting Docker Container')
upAll({ cwd: containerFolder, log: false, commandOptions: ['--build'] }).catch(() => {
	console.error(
		'Failed to start plugin docker environment. Make sure Docker is installed and running.'
	)
})

async function cleanup() {
	try {
		console.log('Shutting Down Docker Container')
		await downAll({ cwd: containerFolder, log: false })
	} catch (e) {
		console.error(e)
	}

	process.exit()
}

// Setup signal handler for Ctrl+C (SIGINT)
process.on('SIGINT', async () => {
	try {
		await cleanup()
	} catch (e) {
		console.error(e)
	}
})

// Maintain the process
setInterval(() => {}, 1000)
