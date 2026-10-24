import { useEffect, useRef } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'
import AccountKey from './AccountKey.tsx'
import DisplayMenu from './DisplayMenu.tsx'
import Sidebar from './Sidebar.tsx'
import VoiceBar from './VoiceBar.tsx'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useUser } from '../auth/useUser.ts'

// The two steps of the way in (sign in, link the phone). After them it is the app: no step bar.
const STEPS: Record<string, number> = { '/login': 1, '/connect-phone': 2 }
const STEP_COUNT = 2

/** The frame around every page: skip link, top bar (name, display settings), how far along
 * the steps you are, and the page. Once someone is signed in it is an app: a sidebar to move around in, and Braillie's voice bar at the
 * foot (speaking or listening, what it says, what it heard). */
const Shell = () => {
  const { user } = useUser()
  const { pathname } = useLocation()
  const main = useRef<HTMLElement>(null)
  const first = useRef(true)
  const step = STEPS[pathname] ?? null
  const done = step === null ? 1 : step / STEP_COUNT
  const { state } = useTutorState(500) // for the navigation and the voice bar

  useEffect(() => {
    // Moving to another page (the address changes without a reload) must read like a new page: it gets its own title in the tab and the
    // screen reader is put at the top of it, on its heading, so it starts reading the new page instead of staying silent on the old one.
    const heading = main.current?.querySelector('h1')
    if (heading) {
      document.title = `${heading.textContent} – Braillie`
      if (!first.current || !document.activeElement || document.activeElement === document.body) {
        heading.setAttribute('tabindex', '-1')
        heading.focus({ preventScroll: true })
      }
    }
    first.current = false
    window.scrollTo(0, 0)
  }, [pathname])

  const inApp = !!user && pathname !== '/login' // signed in and past the welcome: the app around the page
  // home and the activities are Braillo and a few big keys: no sidebar (home is a key) and no voice bar (Braillo shows his own words)
  const bare = pathname === '/modes' || pathname === '/practice' || pathname === '/account'
  const framed = inApp && !bare

  return (
    <>
      <a className="skip" href="#main" onClick={() => main.current?.focus()}>
        Skip to the page
      </a>
      <header className="topbar">
        <Link className="brand" to="/" aria-label="Braillie, home page">
          <span>
            Braillie<span className="brand__dot">.</span>
          </span>
        </Link>
        <div className="topbar__tools">
          {user && pathname !== '/account' && <AccountKey className={framed ? 'topbar__account--app' : undefined} />}
          <DisplayMenu />
        </div>
      </header>
      {step !== null && (
        <div className="progress" aria-hidden="true">
          <span style={{ width: `${Math.round(done * 100)}%` }} />
        </div>
      )}
      <div className={framed ? 'app app--nav' : 'app'}>
        {framed && user && <Sidebar state={state} user={user} />}
        <main id="main" className="stage" tabIndex={-1} ref={main}>
          <Outlet />
        </main>
      </div>
      {framed && state && <VoiceBar state={state} />}
    </>
  )
}

export default Shell
