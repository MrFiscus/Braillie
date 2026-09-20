import { useEffect, useRef, useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import BigButton from '../components/BigButton.tsx'
import BrailleWord from '../components/BrailleWord.tsx'
import { signInWithGoogle } from '../components/GoogleLogin.tsx'
import { googleSignInAvailable } from '../auth/config.ts'
import { useUser } from '../auth/useUser.ts'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useLoginVoice } from '../voice/useLoginVoice.ts'

const SAYING = { speaking: 'Braillie is speaking', listening: 'Listening', idle: 'Voice is off' } as const

/**
 * The first page. Made to be used without seeing it: real headings and labels in a sensible order, one thing to do per section, big targets,
 * strong focus rings, errors announced. And it TALKS: the welcome is spoken and you answer aloud ("Google", or "guest" and then your name).
 * Everything that can be said can also be done with the keyboard, the mouse or a screen reader.
 */
const SignIn = () => {
  const navigate = useNavigate()
  const { user, loading, continueAsGuest, signOut } = useUser()
  const { state, reach } = useTutorState(1500)
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

  return (
    <div className="page page--split">
      <div className="intro">
        <p className="eyebrow">Learn braille by touch</p>
        <h1 ref={heading} tabIndex={-1}>
          Welcome to Braillie
        </h1>
        <p className="lede" id="intro">
          {known
            ? 'Good to see you again. Carry on, or switch to someone else.'
            : 'Braillie teaches you braille with a voice that guides you and a sheet of raised dots under your fingers. Choose how you would like to begin.'}
        </p>
        <section className="voice-block" aria-labelledby="voice-h">
          <h2 id="voice-h" className="sr-only">Talk to Braillie</h2>
          {voice.kind === 'none' ? (
            <p className="hint">
              Speaking to Braillie needs the Braillie tutor running on this computer, or the Chrome, Edge or Safari browser. You can use the buttons below instead.
            </p>
          ) : (
            <div className="voice">
              <div className="voice__row">
                <span className="wave" data-on={voice.status.phase} aria-hidden="true">
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                </span>
                <span className="voice__label" role={voice.running ? 'status' : undefined}>
                  {voice.running ? talkingAbout : 'Voice is off'}
                </span>
                {voice.running ? (
                  <BigButton variant="secondary" quiet onClick={voice.turnOff}>
                    Turn voice off
                  </BigButton>
                ) : (
                  <BigButton variant={voice.kind === 'browser' ? 'primary' : 'secondary'} quiet onClick={voice.start}>
                    {voice.kind === 'tutor' ? 'Talk to Braillie' : 'Use my voice'}
                  </BigButton>
                )}
              </div>
              {voice.running && (
                <>
                  <p className="hint" aria-live="polite">
                    {voice.status.hint}
                  </p>
                  <p className="voice__heard" aria-live="polite">
                    {voice.status.heard ? `I heard: “${voice.status.heard}”` : ''}
                  </p>
                </>
              )}
              {voice.kind === 'browser' && !voice.running && (
                <p className="hint small">Your browser will ask to use the microphone. Choose Allow.</p>
              )}
            </div>
          )}
          {voice.note && (
            <p className="note note--error" role="alert">
              <b>Voice problem.</b> {voice.note}
            </p>
          )}
        </section>

        <BrailleWord word="braillie" caption="Braillie, written in braille." />
      </div>

      <div className="sheet">
        {loading && !user && (
          <p className="hint" role="status">
            Checking who is here…
          </p>
        )}

        {known && user && (
          <section className="sheet__part" aria-labelledby="back-h">
            <h2 id="back-h">Welcome back, {user.name}</h2>
            <p id="back-d">{user.kind === 'google' ? 'Your progress is remembered.' : 'You are continuing without an account, so your progress is not saved.'}</p>
            <BigButton aria-describedby="back-d" onClick={() => navigate('/connect-phone')}>
              Continue as {user.name}
            </BigButton>
            <BigButton variant="secondary" onClick={() => void signOut()}>
              Not {user.name}? Choose another way
            </BigButton>
          </section>
        )}

        {!loading && !user && (
          <>

        <section className="sheet__part" aria-labelledby="google-h">
          <h2 id="google-h">Sign in with Google</h2>
          <p id="google-d">Your progress is remembered, so you can carry on where you left off.</p>
          <BigButton aria-describedby="google-d" onClick={() => { voice.stop(); void google() }}>
            Sign in with Google
          </BigButton>
        </section>

        <section className="sheet__part" aria-labelledby="guest-h">
          <form onSubmit={submit} className="sheet__part">
            <h2 id="guest-h">Continue without an account</h2>
            <p id="guest-d">Your progress is not saved.</p>
            <div className="field">
              <label htmlFor="first-name">Your first name, so I can greet you (optional)</label>
              <input id="first-name" name="first-name" type="text" autoComplete="given-name" value={name} onFocus={voice.stop} onChange={(e) => setName(e.target.value)} />
            </div>
            <BigButton type="submit" variant="secondary" aria-describedby="guest-d">
              Continue without an account
            </BigButton>
          </form>
        </section>
          </>
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

export default SignIn
