import React from 'react'
import ReactDOM from 'react-dom/client'
import Page from './Page'

import './styles.css'

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
	<React.StrictMode>
		<Plugin />
	</React.StrictMode>
)

function Plugin(): JSX.Element {
	return <Page />
}
