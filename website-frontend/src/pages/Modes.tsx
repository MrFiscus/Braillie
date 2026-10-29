import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import BigButton from '../components/BigButton.tsx'
import DotCell from '../components/DotCell.tsx'
import Lyrics from '../components/Lyrics.tsx'
import Mascot from '../components/Mascot.tsx'
import MicButton from '../components/MicButton.tsx'
import { DOTS } from '../braille/alphabet.ts'
import { setMic, setMode, type HubMode } from '../tutor/tutorApi.ts'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useSpeechPosition } from '../voice/useSpeechPosition.ts'
import { useSpokenLine } from '../voice/useSpokenLine.ts'
import { useUser } from '../auth/useUser.ts'

// Each way of using Braillie is drawn as its own first letter in braille: L, R, Q.
const CHOICES: { mode: Exclude<HubMode, 'menu'>; title: string; letter: string; what: string }[] = [
  { mode: 'learn', title: 'Learn', letter: 'l', what: 'The letters A to Z, a few at a time, found by touch.' },
  { mode: 'read', title: 'Read', letter: 'r', what: 'Rest a finger on a word and Braillie reads it aloud.' },
  { mode: 'quiz', title: 'Quiz', letter: 'q', what: 'Braillie names a look-alike letter and you find it.' },
]

const greetingFor = (hour: number) => (hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening')

/**
 * Home: Braillo, big, asking what you would like to do (his words as captions while he speaks) with the mute key under him, and three
 * big keys: Learn, Read, Quiz.
 * You answer by voice ("learn", "read", "quiz") or with a key. Nothing else: the progress and the alphabet are part of Learn.
 */
const Modes = () => {
  const navigate = useNavigate()
  const { user, loading } = useUser()
  const { state, reach } = useTutorState(500) // (often enough to show each spoken sentence as it starts)
  const heading = useRef<HTMLHeadingElement>(null)
  const hub = state?.hub ?? null
  const next = hub?.next_lesson ?? null

  useEffect(() => {
    if (!loading && !user) navigate('/login', { replace: true })
  }, [loading, user, navigate])

  useEffect(() => {
    heading.current?.focus()
  }, [user?.name])

  useEffect(() => {
    if (hub && hub.mode !== 'menu') navigate('/practice', { replace: true }) // chosen by voice or here
  }, [hub, navigate])

  const start = (mode: Exclude<HubMode, 'menu'>) => void setMode(mode)
  const greeting = `${greetingFor(new Date().getHours())}${user?.name ? `, ${user.name}` : ''}`
  const { speaking, speech } = useSpokenLine(state) // (the caption is held a moment after each line, so it does not flicker)
  const spoken = useSpeechPosition(speech, state?.settings?.speech_speed)
  const listening = !speaking && !!state?.heard
  const muted = !!state?.muted
  // what Braillo shows when he is not talking: one short line
  const shortLine =
    reach === 'no' ? 'Start Braillie on the laptop' : hub === null && reach === 'yes' ? 'Ready when you are' : `${greeting}! What shall we do?`

  return (
    <div className="home2">
      <h1 ref={heading} tabIndex={-1} className="sr-only">
        {greeting}
      </h1>
      <p className="sr-only" role="status" aria-live="polite">
        {reach === 'no'
          ? 'I cannot reach the tutor. Start it on the laptop, then this page will update by itself.'
          : 'What would you like to do? Say learn, read or quiz, or choose below.'}
      </p>

      <section className="home2__braillo voicecard" aria-label="Braillo, your tutor">
        <div className="voicecard__stage">
          <Mascot state={speaking ? 'speaking' : listening ? 'listening' : 'idle'} words={spoken.sentence * 1000 + spoken.said} />
          <div className="voicecard__words">
            <div className="voicecard__body">
              {speech ? (
                <Lyrics speech={speech} speed={state?.settings?.speech_speed} />
              ) : (
                <p className="voicecard__hint" aria-hidden="true">
                  {shortLine}
                </p>
              )}
            </div>
          </div>
        </div>
        {reach === 'yes' && (
          <div className="voicecard__foot">
            <MicButton
              state={muted ? 'muted' : 'live'}
              label={muted ? 'Turn the microphone on' : 'Mute the microphone'}
              onClick={() => void setMic(muted).catch(() => {})}
            />
          </div>
        )}
      </section>

      <nav className="home2__choices" aria-label="What to do">
        {CHOICES.map((c) => (
          <button key={c.mode} type="button" className="bigchoice" onClick={() => start(c.mode)} disabled={!hub} aria-describedby={`mode-${c.mode}`}>
            <span className="bigchoice__art" aria-hidden="true">
              <DotCell dots={DOTS[c.letter]} size={72} tone="ink" decorative />
            </span>
            <span className="bigchoice__title">{c.title}</span>
            {c.mode === 'learn' && next && (
              <span className="bigchoice__sub" aria-hidden="true">
                {next.title}
              </span>
            )}
            <span className="sr-only" id={`mode-${c.mode}`}>
              {c.what}
              {c.mode === 'learn' && next ? ` Next: lesson ${next.index} of ${next.of}, ${next.title}.` : ''}
            </span>
            <svg className="bigchoice__go" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path d="M4 12h16m-6-6 6 6-6 6" />
            </svg>
          </button>
        ))}
      </nav>

      {hub === null && reach === 'yes' && (
        <BigButton variant="secondary" onClick={() => navigate('/practice')}>
          Go to the practice screen
        </BigButton>
      )}
    </div>
  )
}

export default Modes
