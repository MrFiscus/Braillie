import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../styles-css/index.css'
import StartPage from './StartPage.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <StartPage />
  </StrictMode>,
)
