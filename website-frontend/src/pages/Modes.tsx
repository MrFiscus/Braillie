import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import BigButton from '../components/BigButton.tsx'
import DotCell from '../components/DotCell.tsx'
import { DOTS } from '../braille/alphabet.ts'
import { setMode, type HubMode } from '../tutor/tutorApi.ts'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useTutorSession } from '../tutor/useTutorSession.ts'
import { useUser } from '../auth/useUser.ts'

// Each way of using Braillie is drawn as its own first letter in braille: L, R, Q.
const CHOICES: { mode: Exclude<HubMode, 'menu'>; title: string; letter: string; what: string }[] = [
  { mode: 'learn', title: 'Learn', letter: 'l', what: 'The letters A to Z, a few at a time, on the alphabet sheet. Braillie teaches each one and you find it by touch.' },
  { mode: 'read', title: 'Read', letter: 'r', what: 'Words. Rest a finger on a word on the words sheet and Braillie reads it aloud.' },
  { mode: 'quiz', title: 'Quiz', letter: 'q', what: 'Look-alike letters, on the look-alikes sheet. Braillie names a letter and you find it.' },
]

/** "What do you want to do today, <name>?" The tutor asks out loud when the phone is linked; you answer by voice (learn, read, quiz) or here. */
const Modes = () => {
  const navigate = useNavigate()
  const { user, loading } = useUser()
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

  return (
    <div className="page">
      <div className="intro">
        <p className="eyebrow">Step 3 of 3</p>
        <h1 ref={heading} tabIndex={-1}>
          What do you want to do today{user ? `, ${user.name}` : ''}?
        </h1>
        <p className="lede" role="status" aria-live="polite">
          {reach === 'no'
            ? 'I cannot reach the tutor. Start it on the laptop, then this page will update by itself.'
            : hub === null && reach === 'yes'
              ? 'The tutor was started in one fixed mode, so there is no choice to make. Go to the practice screen instead.'
              : 'Say learn, read, or quiz, or choose below.'}
        </p>
      </div>

      <ul className="choices">
        {CHOICES.map((c) => (
          <li key={c.mode}>
            <button type="button" className="choice" onClick={() => void setMode(c.mode)} disabled={!hub} aria-describedby={`mode-${c.mode}`}>
              <DotCell dots={DOTS[c.letter]} size={104} tone="accent" decorative />
              <span>
                <span className="choice__title">{c.title}</span>
                <span className="choice__what" id={`mode-${c.mode}`}>
                  {c.what}
                </span>
              </span>
              <span className="choice__go" aria-hidden="true">
                →
              </span>
            </button>
          </li>
        ))}
      </ul>

      {hub === null && reach === 'yes' && (
        <BigButton variant="secondary" onClick={() => navigate('/practice')}>
          Go to the practice screen
        </BigButton>
      )}

      {latest && (
        <p className="hint" aria-live="polite">
          Braillie said: “{latest}”
        </p>
      )}
    </div>
  )
}

export default Modes
