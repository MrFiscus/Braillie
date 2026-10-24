/**
 * The microphone as one big round key, a picture instead of words (its name is there for a screen reader, and as a tooltip).
 * `off`: not listening yet, press to start. `live`: listening, with a ring pulsing out; press to mute. `muted`: crossed out; press to listen again.
 */
export type MicState = 'off' | 'live' | 'muted'

const MicButton = ({ state, label, onClick }: { state: MicState; label: string; onClick: () => void }) => (
  <button type="button" className={`micbtn is-${state}`} aria-label={label} title={label} onClick={onClick}>
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M8.5 21h7" />
      {state === 'muted' && <path d="M4 4l16 16" />}
    </svg>
  </button>
)

export default MicButton
