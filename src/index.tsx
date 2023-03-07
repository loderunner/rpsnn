import * as React from 'react'
import { createRoot } from 'react-dom/client'

import App from './App'

import './main.css'

const container = document.getElementById('app')
if (container === null) {
  throw new Error('could not find #app container element')
}

const root = createRoot(container)
root.render(<App />)
