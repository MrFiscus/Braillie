import AccountKey from './AccountKey.tsx'
import { useLocation, useNavigate } from 'react-router-dom'
import DotCell from './DotCell.tsx'
import { DOTS } from '../braille/alphabet.ts'
import { setMode, type HubMode, type TutorState } from '../tutor/tutorApi.ts'
import type { AppUser } from '../auth/appUser.ts'

const ACTIVITIES: { mode: Exclude<HubMode, 'menu'>; title: string; letter: string }[] = [
  { mode: 'learn', title: 'Learn', letter: 'l' },
  { mode: 'read', title: 'Read', letter: 'r' },
  { mode: 'quiz', title: 'Quiz', letter: 'q' },
]

/** Simple line drawings (not characters, which some fonts do not have), sized to sit beside the braille-cell icons. */
const Icon = ({ name }: { name: 'home' | 'phone' }) => (
  <svg className="navitem__icon" viewBox="0 0 24 24" width="30" height="30" aria-hidden="true" focusable="false">
    {name === 'home' ? (
      <path d="M3 11 12 3l9 8M5.5 9.5V21h13V9.5M10 21v-6h4v6" />
    ) : (
      <>
        <rect x="6.5" y="2.5" width="11" height="19" rx="2.5" />
        <path d="M10.5 18h3" />
      </>
    )}
  </svg>
)

/** The app's navigation once someone is signed in: home, straight into an activity, the phone, and who is here. Each activity is drawn
 * as its own first letter in braille, as on the home screen. A nav landmark with plain buttons: the current one is marked for screen
 * readers (aria-current) and on screen by a solid fill, not by colour alone. */
const Sidebar = ({ state, user }: { state: TutorState | null; user: AppUser }) => {
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const hub = state?.hub ?? null
  const doing = pathname === '/practice' && hub && hub.mode !== 'menu' ? hub.mode : null

  const go = (mode: HubMode) => {
    void setMode(mode)
      .then(() => navigate(mode === 'menu' ? '/modes' : '/practice'))
      .catch(() => navigate('/modes'))
  }

  return (
    <nav className="sidebar" aria-label="Braillie">
      <ul className="sidebar__list">
        <li>
          <button type="button" className="navitem" aria-current={pathname === '/modes' ? 'page' : undefined} onClick={() => go('menu')}>
            <Icon name="home" />
            Home
          </button>
        </li>
        {ACTIVITIES.map((a) => (
          <li key={a.mode}>
            <button type="button" className="navitem" aria-current={doing === a.mode ? 'page' : undefined} disabled={!hub} onClick={() => go(a.mode)}>
              <DotCell dots={DOTS[a.letter]} size={30} tone="accent" decorative />
              {a.title}
            </button>
          </li>
        ))}
        {state?.phone && (
          <li>
            <button type="button" className="navitem" aria-current={pathname === '/connect-phone' ? 'page' : undefined} onClick={() => navigate('/connect-phone')}>
              <Icon name="phone" />
              Phone
            </button>
          </li>
        )}
      </ul>
      <div className="sidebar__who">
        <AccountKey />
        <p>
          <b>{user.name || 'Guest'}</b>
          <span className="hint small">{user.kind === 'google' ? 'Signed in with Google' : 'Guest: progress is not saved'}</span>
        </p>
      </div>
    </nav>
  )
}

export default Sidebar
