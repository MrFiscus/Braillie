import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { getToKnowStyles, themeTokens } from '../styles-react-components/mainStyles.tsx'
import BigButton from '../components/BigButton.tsx'
import { setMode, type HubMode } from '../tutor/tutorApi.ts'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useTutorSession } from '../tutor/useTutorSession.ts'
import { useUser } from '../auth/useUser.ts'

const CHOICES: { mode: Exclude<HubMode, 'menu'>; title: string; what: string }[] = [
  { mode: 'learn', title: 'Learn', what: 'Learn the letters A to Z, a few at a time, with the alphabet sheet. The tutor teaches each letter and you find it by touch.' },
  { mode: 'read', title: 'Read', what: 'Read words. Rest a finger on a word on the words sheet and the tutor reads it aloud.' },
  { mode: 'quiz', title: 'Quiz', what: 'Test yourself on look-alike letters, with the look-alikes sheet. The tutor names a letter and you find it.' },
]

/** "What do you want to do today, <name>?" The tutor asks out loud when the phone is linked; you answer by voice (learn, read, quiz) or here. */
const Modes = () => {
  const styles = getToKnowStyles
  const navigate = useNavigate()
  const { user, loading, signOut } = useUser()
  const { state, reach } = useTutorState(600)
  useTutorSession(user, state)
  const heading = useRef<HTMLHeadingElement>(null)
  const hub = state?.hub ?? null
  const latest = state?.said.length ? state.said[state.said.length - 1].text : ''

  useEffect(() => {
    if (!loading && !user) navigate('/login', { replace: true })
  }, [loading, user, navigate])

  useEffect(() => {
    heading.current?.focus()
  }, [user?.name])

  useEffect(() => {
    if (hub && hub.mode !== 'menu') navigate('/practice', { replace: true }) // chosen by voice or here
  }, [hub, navigate])

  const choose = (mode: HubMode) => {
    void setMode(mode)
  }

  return (
    <main style={styles.container}>
      <div style={{ ...styles.card, maxWidth: 680, alignItems: 'stretch', textAlign: 'left', gap: 22 }}>
        <h1 ref={heading} tabIndex={-1} className="a11y-heading" style={{ ...styles.title, textAlign: 'left', margin: 0 }}>
          What do you want to do today{user ? `, ${user.name}` : ''}?
        </h1>
        <p role="status" aria-live="polite" style={{ margin: 0, fontSize: '1.15rem', color: themeTokens.colors.textSecondary }}>
          {reach === 'no'
            ? 'I cannot reach the tutor. Start it on the laptop, then this page will update by itself.'
            : hub === null && reach === 'yes'
              ? 'The tutor was started in one fixed mode, so there is no choice to make. Go to the practice screen instead.'
              : 'You can say learn, read, or quiz, or choose here.'}
        </p>

        {CHOICES.map((c) => (
          <BigButton key={c.mode} className="a11y-mode" onClick={() => choose(c.mode)} disabled={!hub} aria-describedby={`mode-${c.mode}`}>
            <span>{c.title}</span>
            <small id={`mode-${c.mode}`}>{c.what}</small>
          </BigButton>
        ))}

        {hub === null && reach === 'yes' && (
          <BigButton variant="secondary" onClick={() => navigate('/practice')}>
            Go to the practice screen
          </BigButton>
        )}

        {latest && (
          <p aria-live="polite" style={{ margin: 0, color: themeTokens.colors.textMuted, fontSize: 14 }}>
            The tutor said: {latest}
          </p>
        )}

        <BigButton
          variant="secondary"
          onClick={() => {
            void signOut().then(() => navigate('/login'))
          }}
        >
          {user ? `Not ${user.name}? Sign out` : 'Sign out'}
        </BigButton>
      </div>
    </main>
  )
}

export default Modes
