import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getToKnowStyles, themeTokens } from '../styles-react-components/mainStyles.tsx'
import BigButton from '../components/BigButton.tsx'
import { TUTOR_API, speak } from '../tutor/tutorApi.ts'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useTutorSession } from '../tutor/useTutorSession.ts'
import { useUser } from '../auth/useUser.ts'

// After signing in: link the phone (a QR code to scan; the tutor also SAYS how, for someone who cannot see the screen). The moment the
// phone is linked the tutor greets the user by name and asks what they want to do, and this page moves on to the choice by itself. Started
// without --phone-camera, the laptop's camera is used and there is nothing to link.
const ConnectPhone = () => {
  const styles = getToKnowStyles
  const navigate = useNavigate()
  const { user, loading } = useUser()
  const { state, reach } = useTutorState()
  useTutorSession(user, state)
  const phone = state?.phone ?? null
  const menuDriven = !!state?.hub
  const knowsUser = !!state?.hub?.user || !menuDriven // (a tutor that is not menu-driven has no one to know)
  const linked = reach === 'yes' && (!phone || phone.connected)

  useEffect(() => {
    if (!loading && !user) navigate('/login', { replace: true })
  }, [loading, user, navigate])

  useEffect(() => {
    if (!linked || !user || !knowsUser) return
    const t = window.setTimeout(() => navigate(menuDriven ? '/modes' : '/practice'), 2500) // long enough to hear the start of the greeting
    return () => window.clearTimeout(t)
  }, [linked, user, knowsUser, menuDriven, navigate]) // (plain true/false values: an object that changes with every poll would restart the timer forever)

  let message = 'Looking for the tutor…'
  if (reach === 'no') message = 'I cannot reach the tutor. Start it on the laptop, then this page will update by itself.'
  else if (reach === 'yes' && !phone) message = 'The tutor is using this laptop’s camera, so there is nothing to connect.'
  else if (phone && !phone.connected) message = 'Scan this code with your phone’s camera, then open the link.'
  else if (phone) message = 'Your phone is connected. Listen: the tutor will ask what you want to do.'

  return (
    <main style={styles.container}>
      <div style={{ ...styles.card, maxWidth: 680, alignItems: 'stretch', gap: 22 }}>
        <h1 style={styles.title}>Connect your phone</h1>
        {user && <p style={{ margin: 0, color: themeTokens.colors.textSecondary }}>Hello, {user.name}.</p>}
        <p style={{ ...styles.subtitle, fontSize: '1.2rem' }} role="status" aria-live="polite">
          {message}
        </p>

        {reach === 'no' && (
          <p style={{ ...styles.subtitle, margin: 0 }}>
            On the laptop: <code>python tutor_server.py --phone-camera</code>
          </p>
        )}

        {phone && !phone.connected && (
          <>
            <img
              src={`${TUTOR_API}${phone.qr}`}
              alt="QR code. Scan it with your phone’s camera to connect it."
              width={280}
              height={280}
              style={{ imageRendering: 'pixelated', background: '#fff', borderRadius: 12, alignSelf: 'center' }}
            />
            <p style={{ ...styles.subtitle, margin: 0, fontSize: '1.1rem' }}>
              No camera to scan with? On the phone open <b>{phone.address}</b> and enter the code{' '}
              <b style={{ letterSpacing: '0.2em', fontSize: '1.4em' }}>{phone.code}</b>. If the phone says the connection is not private, choose
              Advanced, then continue. Then tap the big button on the phone.
            </p>
            <BigButton variant="secondary" onClick={() => speak(phone.instructions)}>
              Read the steps aloud
            </BigButton>
            {phone.diagnosis && (
              <p role="status" aria-live="polite" style={{ ...styles.subtitle, margin: 0, color: themeTokens.colors.badgeText }}>
                {phone.diagnosis}
              </p>
            )}
          </>
        )}

        {phone?.connected && (
          <img
            src={`${TUTOR_API}/api/video`}
            alt="Live view from your phone camera"
            style={{ width: '100%', borderRadius: 12, border: `2px solid ${state?.page.ok ? themeTokens.colors.accent : themeTokens.colors.cardBorder}` }}
          />
        )}

        <BigButton onClick={() => navigate(menuDriven ? '/modes' : '/practice')} disabled={!linked || !user}>
          Continue
        </BigButton>
      </div>
    </main>
  )
}

export default ConnectPhone
