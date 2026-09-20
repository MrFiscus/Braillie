import { type MouseEvent } from 'react'
import { getToKnowStyles, themeTokens } from '../styles-react-components/mainStyles.tsx'
import CustomButton from '../styles-react-components/CustomButton.tsx'
import { TUTOR_API, sendCommand, sendFinger } from '../tutor/tutorApi.ts'
import { useTutorState } from '../tutor/useTutorState.ts'

const SHEET_NAMES: Record<string, string> = {
  alphabet: 'Alphabet A to Z',
  words: 'Words',
  numbers: 'Numbers and signs',
  lookalikes: 'Look-alikes',
}
const COMMANDS = ['next page', 'repeat', 'hint', 'found it', 'stop'] // the ones that mean something while exploring a page

// The practice screen: the live camera with a box on every braille cell the tutor detects (red dots = what it sees, green box = locked
// in as read correctly), what it reads, and the same voice commands as buttons. The tutor speaks; this page shows what it is doing.
const Practice = () => {
  const styles = getToKnowStyles
  const { state, reach } = useTutorState(600)

  const reading = state?.reading
  let status = 'Looking for the tutor…'
  if (reach === 'no') status = 'I cannot reach the tutor. Start it on the laptop: python tutor_server.py --mode explore --sheet alphabet --paper'
  else if (state && !state.camera.ok) status = 'The camera is not giving pictures. Check that it is connected and not used by another program.'
  else if (state && reading?.between_pages) status = 'Looking for the next page… show it to the camera.'
  else if (state && !state.page.ok) status = 'I cannot see the page. Hold the whole sheet in view of the camera.'
  else if (state && reading && reading.total > 0 && reading.locked === reading.total) status = `All ${reading.total} cells read. Rest a finger on a cell to hear it.`
  else if (state && reading) status = `Reading the sheet… ${reading.locked} of ${reading.total} cells locked in.`
  else if (state) status = 'Page found.'

  const onVideoClick = (e: MouseEvent<HTMLImageElement>) => {
    const r = e.currentTarget.getBoundingClientRect() // a click stands in for the fingertip if the camera cannot track it
    sendFinger((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height)
  }

  const finger = state?.finger.cell
  const buttons = COMMANDS.filter((c) => state?.config.commands.includes(c))
  const voiceProblems = state?.config.voice && !state.config.voice.ok ? state.config.voice.problems : []
  const said = state ? [...state.said].reverse().slice(0, 5) : []

  return (
    <main style={styles.container}>
      <div style={{ ...styles.card, maxWidth: 980, alignItems: 'stretch' }}>
        <h1 style={styles.title}>Practice</h1>
        <p style={styles.subtitle} role="status" aria-live="polite">
          {status}
        </p>

        {voiceProblems.length > 0 && (
          <div role="alert" style={{ padding: 12, borderRadius: 12, border: `1px solid ${themeTokens.colors.badgeBorder}`, background: themeTokens.colors.badgeBg, textAlign: 'left' }}>
            <b>You will not hear the tutor:</b>
            <ul style={{ margin: '6px 0 0 18px' }}>
              {voiceProblems.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          </div>
        )}

        <img
          src={`${TUTOR_API}/api/video`}
          alt="Live camera view. Each braille cell the tutor detects has a box around it, with the dots it sees in red and the letter it reads above."
          onClick={onVideoClick}
          style={{ width: '100%', borderRadius: 12, cursor: 'crosshair', background: '#000', minHeight: 160, border: `2px solid ${state?.page.ok ? themeTokens.colors.accent : themeTokens.colors.cardBorder}` }}
        />
        <p style={{ margin: 0, fontSize: 13, color: themeTokens.colors.textMuted }}>
          Boxes: green = read correctly and locked in, amber = still reading or not matching the sheet. Click the video where your fingertip is if the
          camera loses it.
        </p>

        <dl style={{ margin: 0, display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 16px', textAlign: 'left', color: themeTokens.colors.textSecondary }}>
          <dt style={{ color: themeTokens.colors.textMuted }}>Sheet</dt>
          <dd style={{ margin: 0 }}>{state?.config.sheet ? SHEET_NAMES[state.config.sheet] ?? state.config.sheet : '–'}</dd>
          <dt style={{ color: themeTokens.colors.textMuted }}>Finger</dt>
          <dd style={{ margin: 0 }}>{finger ? `${finger.label ?? finger.letter ?? 'a cell'}: dots ${finger.dots.join(', ')}` : 'not on a cell'}</dd>
          <dt style={{ color: themeTokens.colors.textMuted }}>Tutor</dt>
          <dd style={{ margin: 0 }}>{state?.tutor.state ?? '–'}</dd>
        </dl>

        <div style={{ display: 'grid', gap: 10, gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))' }}>
          {buttons.map((c) => (
            <CustomButton key={c} buttonText={c.charAt(0).toUpperCase() + c.slice(1)} onClick={() => sendCommand(c)} />
          ))}
        </div>

        {said.length > 0 && (
          <div style={{ textAlign: 'left' }}>
            <h2 style={{ fontSize: 14, textTransform: 'uppercase', letterSpacing: '0.05em', color: themeTokens.colors.textMuted, margin: '0 0 6px' }}>What the tutor said</h2>
            <ul style={{ listStyle: 'none', margin: 0, padding: 0, color: themeTokens.colors.textSecondary }}>
              {said.map((s) => (
                <li key={s.t} style={{ padding: '4px 0', borderBottom: `1px solid ${themeTokens.colors.cardBorder}` }}>
                  {s.text}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </main>
  )
}

export default Practice
