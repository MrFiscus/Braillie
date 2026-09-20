import { useEffect, useRef } from 'react'
import { Link, Outlet, useLocation, useNavigate } from 'react-router-dom'
import DisplayMenu from './DisplayMenu.tsx'
import DotCell from './DotCell.tsx'
import { useUser } from '../auth/useUser.ts'

/** The frame around every page: skip link, masthead (name, who is here, display settings), the page itself, and a footer. */
const Shell = () => {
  const { user, signOut } = useUser()
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const main = useRef<HTMLElement>(null)
  const first = useRef(true)

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

  return (
    <>
      <a className="skip" href="#main" onClick={() => main.current?.focus()}>
        Skip to the page
      </a>
      <header className="masthead">
        <div className="masthead__row">
          <Link className="brand" to="/login" aria-label="Braillie, start again">
            <DotCell dots={[1, 2]} size={38} tone="accent" decorative />
            Braillie
          </Link>
          <div className="masthead__tools">
            {user && (
              <>
                <span className="who">
                  {user.name}
                  <span className="sr-only">, {user.kind === 'google' ? 'signed in with Google' : 'continuing as a guest'}</span>
                </span>
                <button
                  type="button"
                  className="btn btn--quiet"
                  onClick={() => {
                    void signOut().then(() => navigate('/login'))
                  }}
                >
                  Sign out
                </button>
              </>
            )}
            <DisplayMenu />
          </div>
        </div>
      </header>
      <main id="main" className="stage" tabIndex={-1} ref={main}>
        <Outlet />
      </main>
      <footer className="colophon">
        <span>Braillie teaches braille by touch, with a voice that guides you.</span>
        <span>Set in Fraunces and Atkinson Hyperlegible.</span>
      </footer>
    </>
  )
}

export default Shell
