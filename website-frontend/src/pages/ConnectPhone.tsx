import { useNavigate } from 'react-router-dom'
import { getToKnowStyles, themeTokens } from '../styles-react-components/mainStyles.tsx'
import CustomButton from '../styles-react-components/CustomButton.tsx'
import { TUTOR_API, speak } from '../tutor/tutorApi.ts'
import { useTutorState } from '../tutor/useTutorState.ts'

// Step after Get to Know: connect the camera. With the tutor started on --phone-camera, the user scans a QR code with their phone
// (the tutor also SPEAKS how to connect, for someone who cannot see the screen). Started without it, the laptop's camera is used.
const ConnectPhone = () => {
  const styles = getToKnowStyles
  const navigate = useNavigate()
  const { state, reach } = useTutorState()
  const phone = state?.phone ?? null

  let message = 'Looking for the tutor…'
  if (reach === 'no') message = 'I cannot reach the tutor. Start it on the laptop, then this page will update by itself.'
  else if (reach === 'yes' && !phone) message = 'The tutor is using this laptop’s camera, so there is nothing to connect.'
  else if (phone && !phone.connected) message = 'Scan this code with your phone’s camera, then open the link.'
  else if (phone) message = state?.page.ok ? 'Phone connected and the page is in view. You are ready.' : 'Phone connected. Hold it above the page so the whole sheet is in view.'

  const canContinue = reach === 'yes' && (!phone || phone.connected)

  return (
    <main style={styles.container}>
      <div style={{ ...styles.card, maxWidth: 640 }}>
        <h1 style={styles.title}>Connect your camera</h1>
        <p style={styles.subtitle} role="status" aria-live="polite">
          {message}
        </p>

        {reach === 'no' && (
          <p style={{ ...styles.subtitle, margin: 0 }}>
            On the laptop: <code>python tutor_server.py --mode explore --sheet alphabet --paper</code>
            <br />
            To use a phone as the camera, add <code>--phone-camera</code>.
          </p>
        )}

        {phone && !phone.connected && (
          <>
            <img
              src={`${TUTOR_API}${phone.qr}`}
              alt="QR code. Scan it with your phone’s camera to connect it."
              width={260}
              height={260}
              style={{ imageRendering: 'pixelated', background: '#fff', borderRadius: 12 }}
            />
            <p style={{ ...styles.subtitle, margin: 0 }}>
              No camera to scan with? On the phone open <b>{phone.address}</b> and enter the code{' '}
              <b style={{ letterSpacing: '0.2em', fontSize: '1.4em' }}>{phone.code}</b>.
              <br />
              If the phone says the connection is not private, choose Advanced, then continue.
            </p>
            <CustomButton buttonText="Read the steps aloud" onClick={() => speak(phone.instructions)} />
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

        <CustomButton buttonText="Continue" onClick={() => navigate('/practice')} disabled={!canContinue} />
      </div>
    </main>
  )
}

export default ConnectPhone
