import { useEffect, useRef, useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import BigButton from '../components/BigButton.tsx'
import Lyrics from '../components/Lyrics.tsx'
import MicButton from '../components/MicButton.tsx'
import { useSpeechPosition } from '../voice/useSpeechPosition.ts'
import Mascot, { type MascotState } from '../components/Mascot.tsx'
import { signInWithGoogle } from '../components/GoogleLogin.tsx'
import { googleSignInAvailable } from '../auth/config.ts'
import { useUser } from '../auth/useUser.ts'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useLoginVoice } from '../voice/useLoginVoice.ts'

const SAYING = { speaking: 'Braillo is speaking', listening: 'Listening', idle: 'Voice is off' } as const

/**
 * The first page. Made to be used without seeing it: real headings and labels in a sensible order, one thing to do per section, big targets,
 * strong focus rings, errors announced. And it TALKS: the welcome is spoken and you answer aloud ("Google", or "guest" and then your name).
 * Everything that can be said can also be done with the keyboard, the mouse or a screen reader.
 */
const SignIn = () => {
  const navigate = useNavigate()
  const { user, loading, continueAsGuest, signOut } = useUser()
  const { state, reach } = useTutorState(500) // (often enough to show each spoken sentence as it starts)
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const heading = useRef<HTMLHeadingElement>(null)

  useEffect(() => {
    heading.current?.focus() // a screen reader starts by reading the page's title
  }, [])

  const google = async () => {
    setError('')
    if (!googleSignInAvailable()) {
      setError('Google sign-in is not set up on this computer yet. You can still continue without an account.')
      return
    }
    const { error: e } = await signInWithGoogle(`${window.location.origin}/connect-phone`)
    if (e) setError(`Google sign-in did not work: ${e.message}. You can still continue without an account.`)
  }

  const guest = (first: string) => {
    continueAsGuest(first)
    navigate('/connect-phone')
  }

  const voice = useLoginVoice({
    reach,
    state,
    ready: !loading,
    returning: user?.name ?? null,
    actions: {
      google,
      guest: (n) => {
        setName(n)
        guest(n)
      },
      carryOn: () => navigate('/connect-phone'),
      switchAccount: () => signOut(),
    },
  })

  const submit = (e: FormEvent) => {
    e.preventDefault()
    voice.stop() // they have taken over: the voice must not talk over them
    guest(name)
  }
  const known = !loading && !!user
  const wasKnown = useRef(false)
  useEffect(() => {
    if (wasKnown.current && !known) heading.current?.focus() // they chose "another way": start again from the title, so a screen reader reads the options
    wasKnown.current = known
  }, [known])
  const talkingAbout = voice.status.phase === 'idle' ? SAYING.idle : SAYING[voice.status.phase]
  const saying = voice.running && voice.status.phase === 'speaking' ? voice.status.saying : undefined
  // Braillo, the mascot, shows what the voice is doing: talking (its mouth moves with each word), listening (a hand to its ear), or waiting
  const speaking = voice.running && voice.status.phase === 'speaking'
  const spoken = useSpeechPosition(speaking ? state?.speech : null, state?.settings?.speech_speed)
  const braillo: MascotState = speaking ? 'speaking' : voice.running && voice.status.phase === 'listening' ? 'listening' : 'idle'

  return (
    <div className="page page--split signin">
      <div className="intro">
        {/* (no visible title: Braillo and what he says are the welcome; the heading is there for screen readers and the tab title) */}
        <h1 ref={heading} tabIndex={-1} className="sr-only">
          Welcome to Braillie
        </h1>
        <section className="voicecard" aria-labelledby="voice-h" data-on={voice.running ? voice.status.phase : 'off'}>
          <h2 id="voice-h" className="sr-only">
            Talk to Braillo
          </h2>
          <p className="sr-only" role={voice.running ? 'status' : undefined}>
            {voice.kind === 'none' ? 'Voice is not available here' : voice.running ? talkingAbout : 'Voice is off'}
          </p>
          <div className="voicecard__stage">
            <Mascot state={braillo} words={spoken.sentence * 1000 + spoken.said} />
            <div className="voicecard__words">
              <div className="voicecard__body">
                {voice.kind === 'none' ? (
                  <p className="hint">
                    Speaking to Braillie needs the Braillie tutor running on this computer, or the Chrome, Edge or Safari browser. You can use the
                    buttons instead.
                  </p>
                ) : voice.running && voice.status.phase === 'speaking' && state?.speech ? (
                  <Lyrics speech={state.speech} speed={state.settings?.speech_speed} />
                ) : voice.running && saying ? (
                  <p className="voice__saying">
                    <span className="caption__who">Braillo is saying:</span> {saying}
                  </p>
                ) : voice.running ? (
                  <p className="voicecard__hint" aria-live="polite">
                    {voice.status.hint ?? 'Listening.'}
                  </p>
                ) : (
                  <>
                    <p className="voicecard__hint">
                      {known ? (
                        <>
                          Say <b>“continue”</b> or <b>“switch”</b>
                        </>
                      ) : (
                        <>
                          Say <b>“Google”</b> or <b>“guest”</b>
                        </>
                      )}
                    </p>
                    {voice.kind === 'browser' && <p className="hint small">Allow the microphone when your browser asks.</p>}
                  </>
                )}
              </div>

              <p className="voice__heard" aria-live="polite">
                {voice.running && voice.status.heard ? `I heard: “${voice.status.heard}”` : ''}
              </p>
            </div>
          </div>

          <div className="voicecard__foot">
            {voice.kind !== 'none' && (
              <MicButton
                state={voice.running ? 'live' : 'off'}
                label={voice.running ? 'Turn voice off' : voice.kind === 'tutor' ? 'Talk to Braillo' : 'Use my voice'}
                onClick={voice.running ? voice.turnOff : voice.start}
              />
            )}
          </div>
          {voice.note && (
            <p className="note note--error" role="alert">
              <b>Voice problem.</b> {voice.note}
            </p>
          )}
        </section>
      </div>

      <div className="options">
        {loading && !user && (
          <p className="hint" role="status">
            Checking who is here…
          </p>
        )}

        {known && user && (
          <section className="back" aria-labelledby="back-h">
            {/* one choice to make, so it takes the room: a big greeting and two big keys */}
            <h2 id="back-h">
              Welcome back, <span>{user.name}</span>
            </h2>
            <p id="back-d" className="sr-only">
              {user.kind === 'google' ? 'Your progress is remembered.' : 'You are continuing without an account, so your progress is not saved.'}
            </p>
            <BigButton className="back__go" aria-describedby="back-d" onClick={() => navigate('/connect-phone')}>
              Continue as {user.name}
            </BigButton>
            <BigButton className="back__go" variant="secondary" onClick={() => void signOut()}>
              Not {user.name}? Choose another way
            </BigButton>
          </section>
        )}

        {!loading && !user && (
          <section className="start" aria-label="Ways in">
            {/* the same shape as "welcome back": a big greeting, then big keys, each with its picture inside it */}
            <p className="start__hello" aria-hidden="true">
              Let’s <span>start</span>
            </p>
            <div className="start__way" role="group" aria-labelledby="google-h">
              <h2 id="google-h" className="sr-only">
                Sign in with Google
              </h2>
              <p id="google-d" className="sr-only">
                Your progress is remembered, so you can carry on where you left off, on this computer or another.
              </p>
              <BigButton
                className="start__go"
                aria-describedby="google-d"
                onClick={() => {
                  voice.stop()
                  void google()
                }}
              >
                <OptionIcon name="account" />
                Sign in with Google
              </BigButton>
            </div>

            <p className="options__or" aria-hidden="true">
              <span>or</span>
            </p>

            <form onSubmit={submit} className="start__way" aria-labelledby="guest-h">
              <h2 id="guest-h" className="sr-only">
                Continue without an account
              </h2>
              <p id="guest-d" className="sr-only">
                Start straight away. Your progress is not saved.
              </p>
              <div className="field start__field">
                <label htmlFor="first-name">
                  Your name <span className="option__optional">(optional)</span>
                </label>
                <input id="first-name" name="first-name" type="text" autoComplete="given-name" value={name} onFocus={voice.stop} onChange={(e) => setName(e.target.value)} />
              </div>
              <BigButton className="start__go" type="submit" variant="secondary" aria-describedby="guest-d">
                <OptionIcon name="person" />
                Continue as guest
              </BigButton>
            </form>
          </section>
        )}

        <div role="alert">
          {error && (
            <p className="note note--error">
              <b>Problem.</b> {error}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

/** A simple line picture for each way in (not characters, which some fonts lack). */
const OptionIcon = ({ name }: { name: 'account' | 'person' }) => (
  <span className="option__icon" aria-hidden="true">
    <svg viewBox="0 0 24 24" focusable="false">
      {name === 'account' ? (
        <>
          <rect x="4" y="10" width="16" height="11" rx="2.5" />
          <path d="M8 10V7a4 4 0 0 1 8 0v3M12 14.5v2.5" />
        </>
      ) : (
        <>
          <circle cx="12" cy="8" r="4" />
          <path d="M4.5 21c.8-4 3.8-6.5 7.5-6.5s6.7 2.5 7.5 6.5" />
        </>
      )}
    </svg>
  </span>
)

export default SignIn
