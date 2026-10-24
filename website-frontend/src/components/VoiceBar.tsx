import { useEffect, useRef, useState } from 'react'
import Lyrics from './Lyrics.tsx'
import { sendCommand, type TutorState } from '../tutor/tutorApi.ts'

const HEARD_FOR_MS = 8000 // what the microphone made out stays on the bar this long

/**
 * Braillie's voice, always in view at the foot of the app: whether it is speaking or listening (or cannot listen), what it is saying,
 * what it last heard you say, and two keys, Repeat and Quiet. Not a live region: someone using a screen reader already hears Braillie,
 * and the same words read out again over its voice would make both hard to follow; the bar is there to look at, or move to, any time.
 */
const VoiceBar = ({ state }: { state: TutorState }) => {
  const heard = state.heard ?? null
  const [fresh, setFresh] = useState(false)
  const lastHeard = useRef<number | null>(null)

  useEffect(() => {
    if (!heard) return
    const isNew = lastHeard.current !== null && heard.n !== lastHeard.current && !!heard.text
    lastHeard.current = heard.n
    if (!isNew) return
    setFresh(true)
    const t = window.setTimeout(() => setFresh(false), HEARD_FOR_MS)
    return () => window.clearTimeout(t)
  }, [heard])

  const listening = heard !== null // the tutor has a live microphone
  const on = state.speaking ? 'speaking' : listening ? 'listening' : 'off'
  const said = state.said.length ? state.said[state.said.length - 1].text : ''

  return (
    <section className="voicebar" aria-label="Braillo, your tutor" data-on={on}>
      <div className="voicebar__state">
        <span className="wave" data-on={on === 'off' ? 'idle' : on} aria-hidden="true">
          <i />
          <i />
          <i />
          <i />
          <i />
        </span>
        <span className="voicebar__label">{on === 'speaking' ? 'Speaking' : on === 'listening' ? 'Listening' : 'Mic off'}</span>
      </div>
      <div className="voicebar__words">
        {state.speaking && state.speech ? (
          <Lyrics speech={state.speech} speed={state.settings?.speech_speed} />
        ) : (
          said && (
            <p className="voicebar__said">
              <span className="voicebar__who">Braillo said:</span> {said}
            </p>
          )
        )}
        {fresh && heard?.text ? (
          <p className="voicebar__heard">
            <span className="voicebar__who">You:</span> “{heard.text}”
          </p>
        ) : (
          !said && <p className="voicebar__heard hint">Say “help” to hear what you can say, or “Braillo” and ask anything.</p>
        )}
      </div>
      <div className="voicebar__keys">
        <button type="button" className="btn btn--quiet" onClick={() => sendCommand('repeat')}>
          Repeat
        </button>
        <button type="button" className="btn btn--quiet" onClick={() => sendCommand('hush')} disabled={!state.speaking}>
          Quiet
        </button>
      </div>
    </section>
  )
}

export default VoiceBar
