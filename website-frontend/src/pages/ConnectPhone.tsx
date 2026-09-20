import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import BigButton from '../components/BigButton.tsx'
import { TUTOR_API, speak } from '../tutor/tutorApi.ts'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useTutorSession } from '../tutor/useTutorSession.ts'
import { useUser } from '../auth/useUser.ts'

// After signing in: link the phone (a QR code to scan; the tutor also SAYS how, for someone who cannot see the screen). The moment the
// phone is linked the tutor greets the user by name and asks what they want to do, and this page moves on to the choice by itself. Started
// without --phone-camera, the laptop's camera is used and there is nothing to link.
const ConnectPhone = () => {
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
  else if (phone && !phone.connected) message = 'Scan the code with your phone’s camera, then open the link.'
  else if (phone) message = 'Your phone is connected. Listen: the tutor will ask what you want to do.'

  return (
    <div className="page page--split">
      <div className="intro">
        <p className="eyebrow">Step 2 of 3</p>
        <h1>Connect your phone</h1>
        <p className="lede">
          Your phone is Braillie’s eyes and ears: it watches the sheet and hears you, while the laptop does the thinking.
          {user ? ` Hello, ${user.name}.` : ''}
        </p>
        <p className="status" role="status" aria-live="polite">
          {message}
        </p>
        {reach === 'no' && (
          <p>
            On the laptop, run <code>python tutor_server.py --phone-camera</code>
          </p>
        )}
        {phone && !phone.connected && (
          <ol className="steps">
            <li>
              <span>Open the camera on your phone and point it at the code.</span>
            </li>
            <li>
              <span>Tap the link that appears. If the phone says the connection is not private, choose Advanced, then continue.</span>
            </li>
            <li>
              <span>Tap the big button on the phone. Then listen: Braillie will greet you.</span>
            </li>
          </ol>
        )}
      </div>

      <div className="sheet">
        {phone && !phone.connected && (
          <>
            <div className="qr">
              <img src={`${TUTOR_API}${phone.qr}`} alt="QR code. Scan it with your phone’s camera to connect it." width={280} height={280} />
            </div>
            <p>
              No camera to scan with? On the phone open <b>{phone.address}</b> and enter the code <span className="code">{phone.code}</span>
            </p>
            <div className="pair">
              <BigButton variant="secondary" onClick={() => speak(phone.instructions)}>
                Read the steps aloud
              </BigButton>
              <BigButton onClick={() => navigate(menuDriven ? '/modes' : '/practice')} disabled={!linked || !user}>
                Continue
              </BigButton>
            </div>
            {phone.diagnosis && (
              <p className="note note--error" role="status" aria-live="polite">
                <b>Not yet.</b> {phone.diagnosis}
              </p>
            )}
          </>
        )}

        {phone?.connected && (
          <div className="live">
            <img src={`${TUTOR_API}/api/video`} alt="Live view from your phone camera" />
          </div>
        )}

        {!phone && (
          <p className="hint">
            {reach === 'no' ? 'Nothing to show until the tutor is running.' : 'Nothing to connect: the laptop’s own camera is in use.'}
          </p>
        )}

        {!(phone && !phone.connected) && (
          <BigButton onClick={() => navigate(menuDriven ? '/modes' : '/practice')} disabled={!linked || !user}>
            Continue
          </BigButton>
        )}
      </div>
    </div>
  )
}

export default ConnectPhone
