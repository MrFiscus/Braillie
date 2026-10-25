import { Navigate, useNavigate } from 'react-router-dom'
import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import BigButton from '../components/BigButton.tsx'
import Lyrics from '../components/Lyrics.tsx'
import Mascot from '../components/Mascot.tsx'
import MicButton from '../components/MicButton.tsx'
import { sayPrompt, setMic, setMode } from '../tutor/tutorApi.ts'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useAccountVoice } from '../voice/useAccountVoice.ts'
import { useSpeechPosition } from '../voice/useSpeechPosition.ts'
import { useSpokenLine } from '../voice/useSpokenLine.ts'
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

const COUNT_MS = 900

/** A number that counts up to `to` when it first shows (and on to a new value when it changes); straight there for anyone who has asked
 * their device for less motion. A picture of the number: the page gives a screen reader the number itself. */
const useCountUp = (to: number) => {
  const [n, setN] = useState(0)
  const at = useRef(0) // where the count is now, to go on from when the number changes
  useEffect(() => {
    const still = typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (still) {
      at.current = to
      setN(to)
      return
    }
    const from = at.current
    const start = performance.now()
    let frame = 0
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / COUNT_MS)
      at.current = Math.round(from + (to - from) * (1 - (1 - t) ** 3))
      setN(at.current)
      if (t < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [to])
  return n
}

const Stat = ({ icon, value, of, label }: { icon: string; value: number; of?: string; label: string }) => {
  const shown = useCountUp(value)
  return (
    <li>
      <span className="account__icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" focusable="false">
          {ICONS[icon]}
        </svg>
      </span>
      <span className="account__value" aria-hidden="true">
        {shown}
        {of && <small>{of}</small>}
      </span>
      <span className="account__label">
        <span className="sr-only">
          {value}
          {of && ` of ${of.slice(1)}`}{' '}
        </span>
        {label}
      </span>
    </li>
  )
}

/**
 * Your page: who you are and whether your progress is kept; Braillo, who reads out how you are getting on and answers "letters",
 * "continue", "learn", "sign out" and the like (his words as captions beside him, the mute key there too); three big numbers, each
 * with a picture; and your alphabet, each letter with a bar under it that fills as you get to know it (a letter you know fills in
 * whole). Then carry on, or sign out.
 */
const Account = () => {
  const { user, loading, signOut } = useUser()
  const navigate = useNavigate()
  const { state, reach } = useTutorState(500) // (often enough to show each spoken sentence as it starts)
  const progress = state?.progress ?? null
  const leave = () => void signOut().then(() => navigate('/'))
  useAccountVoice({
    reach,
    state,
    name: user?.name ?? null,
    act: (ask) => {
      if (ask === 'home') navigate('/modes')
      else if (ask === 'signout') void sayPrompt('account_bye', user?.name, true).catch(() => {}).finally(leave)
      else void setMode(ask).then(() => navigate('/practice')).catch(() => navigate('/modes'))
    },
  })
  const { speaking, speech } = useSpokenLine(state) // (the caption is held a moment after each line, so it does not flicker)
  const spoken = useSpeechPosition(speech, state?.settings?.speech_speed)
  const listening = !speaking && !!state?.heard
  const muted = !!state?.muted

  if (!loading && !user) return <Navigate to="/login" replace />
  if (!user) return null

  const stats = [
    { icon: 'letters', value: progress?.learned ?? 0, of: '/26', label: 'letters known' },
    { icon: 'streak', value: progress?.streak ?? 0, label: (progress?.streak ?? 0) === 1 ? 'day in a row' : 'days in a row' },
    { icon: 'sessions', value: progress?.sessions ?? 0, label: (progress?.sessions ?? 0) === 1 ? 'session' : 'sessions' },
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

      <section className="account__voice" aria-label="Braillo, your tutor">
        <Mascot state={speaking ? 'speaking' : listening ? 'listening' : 'idle'} size={150} words={spoken.sentence * 1000 + spoken.said} />
        <div className="account__words">
          {speech ? (
            <Lyrics speech={speech} speed={state?.settings?.speech_speed} />
          ) : (
            <p className="account__hint" aria-hidden="true">
              {reach === 'no' ? 'Start Braillie on the laptop to talk to me.' : 'Say “letters”, “continue” or “sign out”.'}
            </p>
          )}
        </div>
        {reach === 'yes' && (
          <MicButton
            state={muted ? 'muted' : 'live'}
            label={muted ? 'Turn the microphone on' : 'Mute the microphone'}
            onClick={() => void setMic(muted).catch(() => {})}
          />
        )}
      </section>

      {progress ? (
        <>
          <ul className="account__stats" aria-label="How you are getting on">
            {stats.map((s) => (
              <Stat key={s.label} {...s} />
            ))}
          </ul>

          <section aria-labelledby="alpha-h" className="account__alpha">
            <h2 id="alpha-h">Your alphabet</h2>
            <ul className="alpha">
              {ALPHABET.map((l, n) => {
                const practised = l in progress.mastery
                const known = progress.mastery[l] ?? 0
                const level = !practised ? 'none' : Math.round(known * 4) === 4 ? 'known' : 'some'
                return (
                  <li key={l} data-level={level} style={{ '--known': known, '--n': n } as CSSProperties}>
                    <span className="alpha__letter" aria-hidden="true">
                      {l.toUpperCase()}
                    </span>
                    <span className="alpha__bar" aria-hidden="true">
                      <i />
                    </span>
                    <span className="sr-only">
                      {l.toUpperCase()}: {practised ? `${Math.round(known * 100)} percent known` : 'not practised yet'}
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
        <BigButton onClick={() => navigate('/modes')}>Continue</BigButton>
        <BigButton variant="secondary" className="btn--line" onClick={leave}>
          Sign out
        </BigButton>
      </div>
    </div>
  )
}

export default Account
