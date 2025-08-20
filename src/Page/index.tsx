import { useEffect, useState } from 'react'
import styles from './styles.module.css'

// NOTE: Change the PORT to the port of the server from the backend
// The app's server is running on port 8000, so use a different port
// that is not being used by the app or any other plugins
const PORT = 8001

function TemplatePage(): JSX.Element {
	const [testResult, setTestResult] = useState<string>('')

	useEffect(() => {
		const runFetch = () => {
			fetch(`http://localhost:${PORT}/`)
				.then((res) => res.text())
				.then((data) => setTestResult(data))
				.catch(() => setTimeout(runFetch, 1000))
		}

		if (testResult === '') {
			runFetch()
		}
	}, [])

	useEffect(() => {
		const registerMessage = {
			type: 'register-plugin',
			data: {
				pluginName: 'plugin-template'
			}
		}

		// Register the plugin with the main app
		// Doing so allows the main app to send messages to the plugin,
		// which it primarily uses to send directory information (i.e. the open folder in the main app)
		parent.postMessage(registerMessage, '*')

		// Listen for messages from the main app
		// See iframe.tsx and iframe-message-schema.ts in the main app for more information
		// You can use this to get directory information from the main app,
		// to read file contents, or to save file contents through the main app
		const listener = (event: MessageEvent) => {
			console.log(event.data)
		}

		window.addEventListener('message', listener)

		return () => {
			window.removeEventListener('message', listener)
		}
	}, [])

	return (
		<div>
			<h1 className={styles.header}>{testResult === '' ? 'Template Page!' : testResult}</h1>
		</div>
	)
}

export default TemplatePage
