import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import BigButton from '../components/BigButton.tsx'
import Lyrics from '../components/Lyrics.tsx'
import Mascot from '../components/Mascot.tsx'
import MicButton from '../components/MicButton.tsx'
import { useSpeechPosition } from '../voice/useSpeechPosition.ts'
import { TUTOR_API, sayPrompt, setMic, shareTheme } from '../tutor/tutorApi.ts'
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

  // On arrival, Braillie says how to connect the phone (in its own voice, with the captions), once: someone who cannot see the QR
  // code hears what to do, and how to type the code instead.
  const told = useRef(false)
  const waiting = reach === 'yes' && !!phone && !phone.connected && !!user
  useEffect(() => {
    if (!waiting || told.current) return
    told.current = true
    void sayPrompt('connect_phone', undefined, true).catch(() => {})
  }, [waiting])

  // (the tutor may have started after this page did: tell it the colours again when the phone joins, so the phone's page matches)
  const phoneOn = !!phone?.connected
  useEffect(() => {
    if (phoneOn) shareTheme(document.documentElement.dataset.theme)
  }, [phoneOn])

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

  const next = () => navigate(menuDriven ? '/modes' : '/practice')
  // what Braillo says on screen when it is not talking: one short line
  const shortLine =
    reach === 'no' ? 'Start Braillie on the laptop' : !phone ? 'Nothing to connect' : phone.connected ? 'Phone connected!' : 'Scan the code with your phone'
  const speaking = !!state?.speaking && !!state?.speech
  const spoken = useSpeechPosition(speaking ? state?.speech : null, state?.settings?.speech_speed)

  const muted = !!state?.muted
  const again = () => void sayPrompt('connect_phone', undefined, true).catch(() => {})

  // Laid out like home: Braillo on the left saying what to do (the mute key and "hear it again" under him), and on the right the one
  // thing to use, the code to scan (then the phone's own view once it is linked).
  return (
    <div className="home2 connect2">
      <h1 className="sr-only">Connect your phone</h1>
      <p className="sr-only" role="status" aria-live="polite">
        {message}
      </p>

      <section className="home2__braillo voicecard" aria-label="Braillo, your tutor">
        <div className="voicecard__stage">
          <Mascot state={speaking ? 'speaking' : 'idle'} words={spoken.sentence * 1000 + spoken.said} />
          <div className="voicecard__words">
            <div className="voicecard__body">
              {speaking && state?.speech ? (
                <Lyrics speech={state.speech} speed={state.settings?.speech_speed} />
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
            {phone && !phone.connected && (
              <button type="button" className="roundkey" aria-label="Hear the steps again" title="Hear the steps again" onClick={again}>
                <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                  <path d="M4 12a8 8 0 1 0 2.5-5.8M4 4v4.5h4.5" />
                </svg>
              </button>
            )}
          </div>
        )}
        {reach === 'no' && (
          <p>
            <code>python tutor_server.py --phone-camera</code>
          </p>
        )}
      </section>

      <div className="connect2__side">
        {phone && !phone.connected && (
          <>
            <div className="qr qr--big">
              <img src={`${TUTOR_API}${phone.qr}`} alt="QR code. Scan it with your phone’s camera to connect it." width={420} height={420} />
            </div>
            <p className="connect__manual">
              No camera? Open <b>{phone.address}</b>, code <span className="code">{phone.code}</span>
            </p>
            {(phone.diagnosis || !phone.trusted) && (
              <details className="trouble">
                <summary>Trouble connecting?</summary>
                {!phone.trusted && <p>If the phone says the connection is not private, choose Advanced, then continue.</p>}
                {phone.diagnosis && (
                  <p role="status" aria-live="polite">
                    {phone.diagnosis}
                  </p>
                )}
              </details>
            )}
          </>
        )}

        {linked && (
          <>
            {phone?.connected && (
              <div className="live">
                <img src={`${TUTOR_API}/api/video`} alt="Live view from your phone camera" />
              </div>
            )}
            <BigButton className="connect2__go" onClick={next} disabled={!user}>
              Continue
            </BigButton>
          </>
        )}
      </div>
    </div>
  )
}

export default ConnectPhone
