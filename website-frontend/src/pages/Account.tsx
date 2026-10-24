import { Navigate, useNavigate } from 'react-router-dom'
import type { ReactNode } from 'react'
import BigButton from '../components/BigButton.tsx'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useUser } from '../auth/useUser.ts'

const ALPHABET = 'abcdefghijklmnopqrstuvwxyz'.split('')

// a picture for each number: a star (letters known), a flame (days in a row), a calendar (sessions)
const ICONS: Record<string, ReactNode> = {
  letters: <path d="M12 3l2.6 5.6 6.1.7-4.5 4.2 1.2 6L12 16.6 6.6 19.5l1.2-6L3.3 9.3l6.1-.7z" />,
  streak: <path d="M12 21c-4 0-6.5-2.6-6.5-6.2 0-3.4 2.4-5.4 3.6-8.3.6 1.6 1.5 2.6 2.4 3.1.4-2.9 1.8-5.1 4-6.6-.3 3 1.1 5 2.2 6.8 1 1.6 1.8 3 1.8 5 0 3.6-2.5 6.2-7.5 6.2z" />,
  sessions: (
    <>
      <rect x="3.5" y="5" width="17" height="15" rx="2.5" />
      <path d="M3.5 10h17M8 3v4M16 3v4" />
    </>
  ),
}

/**
 * Your page: who you are and whether your progress is kept, how you are getting on (three big numbers, each with a picture), and your
 * alphabet, each letter
 * with four dots under it that fill in as you get to know it. Then carry on, or sign out.
 */
const Account = () => {
  const { user, loading, signOut } = useUser()
  const navigate = useNavigate()
  const { state, reach } = useTutorState(2000)
  const progress = state?.progress ?? null

  if (!loading && !user) return <Navigate to="/login" replace />
  if (!user) return null

  const stats = [
    { icon: 'letters', value: `${progress?.learned ?? 0}`, of: '/26', label: 'letters known' },
    { icon: 'streak', value: `${progress?.streak ?? 0}`, label: (progress?.streak ?? 0) === 1 ? 'day in a row' : 'days in a row' },
    { icon: 'sessions', value: `${progress?.sessions ?? 0}`, label: (progress?.sessions ?? 0) === 1 ? 'session' : 'sessions' },
  ]

  return (
    <div className="account">
      <header className="account__head">
        <span className="account__avatar" aria-hidden="true">
          {user.name.charAt(0).toUpperCase()}
        </span>
        <div>
          <h1>Hi, {user.name}</h1>
          <p className="account__kind">
            {user.kind === 'google' ? 'Signed in with Google. Your progress is saved.' : 'Guest. Your progress is not saved.'}
          </p>
        </div>
      </header>

      {progress ? (
        <>
          <ul className="account__stats" aria-label="How you are getting on">
            {stats.map((s) => (
              <li key={s.label}>
                <span className="account__icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24" focusable="false">
                    {ICONS[s.icon]}
                  </svg>
                </span>
                <span className="account__value">
                  {s.value}
                  {s.of && <small>{s.of}</small>}
                </span>
                <span className="account__label">{s.label}</span>
              </li>
            ))}
          </ul>

          <section aria-labelledby="alpha-h" className="account__alpha">
            <h2 id="alpha-h">Your alphabet</h2>
            <ul className="alpha">
              {ALPHABET.map((l) => {
                const practised = l in progress.mastery
                const pips = Math.round((progress.mastery[l] ?? 0) * 4)
                return (
                  <li key={l} data-none={!practised}>
                    <span className="alpha__letter" aria-hidden="true">
                      {l.toUpperCase()}
                    </span>
                    <span className="alpha__pips" aria-hidden="true">
                      {[0, 1, 2, 3].map((i) => (
                        <i key={i} data-on={i < pips} />
                      ))}
                    </span>
                    <span className="sr-only">
                      {l.toUpperCase()}: {practised ? `${Math.round((progress.mastery[l] ?? 0) * 100)} percent known` : 'not practised yet'}
                    </span>
                  </li>
                )
              })}
            </ul>
          </section>
        </>
      ) : (
        <p className="account__none">
          {reach === 'no' ? 'Start Braillie on the laptop to see how you are getting on.' : 'Your progress shows here once you have had a lesson.'}
        </p>
      )}

      <div className="account__keys">
        <BigButton onClick={() => navigate('/modes')}>
          <svg className="btn__icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M4 12h16m-6-6 6 6-6 6" />
          </svg>
          Continue
        </BigButton>
        <BigButton variant="secondary" onClick={() => void signOut().then(() => navigate('/'))}>
          <svg className="btn__icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M14 4h4a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-4M9 16l-4-4 4-4M5 12h10" />
          </svg>
          Sign out
        </BigButton>
      </div>
    </div>
  )
}

export default Account
