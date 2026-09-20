import { useEffect, useRef, useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { getToKnowStyles } from '../styles-react-components/mainStyles.tsx'
import BigButton from '../components/BigButton.tsx'
import { signInWithGoogle } from '../components/GoogleLogin.tsx'
import { googleSignInAvailable } from '../auth/config.ts'
import { useUser } from '../auth/useUser.ts'
import { sayPrompt } from '../tutor/tutorApi.ts'
import { useTutorState } from '../tutor/useTutorState.ts'

const WELCOMED = 'braillie.welcomed'

/**
 * The first page. Made to be used without seeing it: real headings and labels in a sensible order (Google first), one thing to do per
 * section, big targets, strong focus rings, errors announced, and the tutor SAYS the options aloud when it is running.
 */
const SignIn = () => {
  const styles = getToKnowStyles
  const navigate = useNavigate()
  const { user, loading, continueAsGuest, signOut } = useUser()
  const { reach } = useTutorState(1500)
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const heading = useRef<HTMLHeadingElement>(null)

  useEffect(() => {
    heading.current?.focus() // a screen reader starts by reading the page's title
  }, [])

  useEffect(() => {
    if (reach !== 'yes') return
    try {
      if (sessionStorage.getItem(WELCOMED)) return
      sessionStorage.setItem(WELCOMED, '1')
    } catch {
      // no storage: it may be said again on a refresh, which is harmless
    }
    sayPrompt('welcome').catch(() => {})
  }, [reach])

  const google = async () => {
    setError('')
    if (!googleSignInAvailable()) {
      setError('Google sign-in is not set up on this computer yet. You can still continue without an account.')
      return
    }
    const { error: e } = await signInWithGoogle(`${window.location.origin}/connect-phone`)
    if (e) setError(`Google sign-in did not work: ${e.message}. You can still continue without an account.`)
  }

  const guest = (e: FormEvent) => {
    e.preventDefault()
    continueAsGuest(name)
    navigate('/connect-phone')
  }

  return (
    <main style={styles.container}>
      <div style={{ ...styles.card, maxWidth: 680, alignItems: 'stretch', textAlign: 'left', gap: 28 }}>
        <div>
          <h1 ref={heading} tabIndex={-1} className="a11y-heading" style={{ ...styles.title, textAlign: 'left', margin: 0 }}>
            Welcome to Braillie
          </h1>
          <p id="intro" style={{ ...styles.subtitle, textAlign: 'left', fontSize: '1.2rem', marginTop: 12 }}>
            Braillie teaches you braille by touch, with a voice that guides you. Choose how you would like to continue.
          </p>
        </div>

        {!loading && user && (
          <section aria-labelledby="back-h" style={{ display: 'grid', gap: 12 }}>
            <h2 id="back-h" style={{ margin: 0, fontSize: '1.5rem' }}>Welcome back, {user.name}</h2>
            <p id="back-d" style={{ margin: 0, fontSize: '1.15rem' }}>
              {user.kind === 'google' ? 'Your progress is remembered.' : 'You are continuing without an account, so your progress is not saved.'}
            </p>
            <BigButton aria-describedby="back-d" onClick={() => navigate('/connect-phone')}>
              Continue as {user.name}
            </BigButton>
            <BigButton variant="secondary" onClick={() => void signOut()}>
              Not {user.name}? Choose another way
            </BigButton>
          </section>
        )}

        <section aria-labelledby="google-h" style={{ display: 'grid', gap: 12 }}>
          <h2 id="google-h" style={{ margin: 0, fontSize: '1.5rem' }}>Sign in with Google</h2>
          <p id="google-d" style={{ margin: 0, fontSize: '1.15rem' }}>Your progress is remembered, so you can carry on where you left off.</p>
          <BigButton aria-describedby="google-d" onClick={() => void google()}>
            Sign in with Google
          </BigButton>
        </section>

        <section aria-labelledby="guest-h">
          <form onSubmit={guest} style={{ display: 'grid', gap: 12 }}>
            <h2 id="guest-h" style={{ margin: 0, fontSize: '1.5rem' }}>Continue without an account</h2>
            <p id="guest-d" style={{ margin: 0, fontSize: '1.15rem' }}>Your progress is not saved.</p>
            <label htmlFor="first-name" style={{ fontSize: '1.15rem', fontWeight: 600 }}>
              Your first name, so I can greet you (optional)
            </label>
            <input id="first-name" className="a11y-input" name="first-name" type="text" autoComplete="given-name" value={name} onChange={(e) => setName(e.target.value)} />
            <BigButton type="submit" variant="secondary" aria-describedby="guest-d">
              Continue without an account
            </BigButton>
          </form>
        </section>

        <div role="alert" style={{ minHeight: '1.5rem', fontSize: '1.15rem', fontWeight: 600, color: '#fca5a5' }}>
          {error}
        </div>
      </div>
    </main>
  )
}

export default SignIn
