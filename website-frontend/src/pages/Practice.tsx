import { type MouseEvent } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { getToKnowStyles, themeTokens } from '../styles-react-components/mainStyles.tsx'
import CustomButton from '../styles-react-components/CustomButton.tsx'
import { TUTOR_API, sendCommand, sendFinger, setMode } from '../tutor/tutorApi.ts'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useProgressSync } from '../tutor/useProgressSync.ts'
import { useTutorSession } from '../tutor/useTutorSession.ts'
import { useUser } from '../auth/useUser.ts'
import LearnPanel from '../components/LearnPanel.tsx'
import { QuizPanel, ReadPanel } from '../components/ActivityPanels.tsx'

const SHEET_NAMES: Record<string, string> = {
  alphabet: 'Alphabet A to Z',
  words: 'Words',
  numbers: 'Numbers and signs',
  lookalikes: 'Look-alikes',
}
const COMMANDS = ['next page', 'repeat', 'hint', 'found it', 'stop'] // the ones that mean something while exploring a page
const LEARN_COMMANDS = ['start quiz', 'repeat', 'hint', 'found it', 'next', 'explore', 'practice', 'next page', 'stop'] // in the guided lessons
const QUIZ_COMMANDS = ['repeat', 'hint', 'next', 'stop']
const READ_COMMANDS = ['repeat', 'stop']
const LABELS: Record<string, string> = { 'start quiz': 'Start / continue', 'found it': 'I found it', explore: 'Free explore', practice: 'Practice review', stop: 'Finish' }
const TITLES: Record<string, string> = { learn: 'Learn braille', read: 'Read', quiz: 'Quiz' }

// The practice screen: the live camera with a box on every braille cell the tutor detects (red dots = what it sees, green box = locked
// in as read correctly), what it reads, and the same voice commands as buttons. The tutor speaks; this page shows what it is doing.
const Practice = () => {
  const styles = getToKnowStyles
  const { state, reach } = useTutorState(600)
  const navigate = useNavigate()
  const { user, loading } = useUser()
  useTutorSession(user, state)
  const sync = useProgressSync(state, user?.kind === 'google')
  const learning = state?.learning ?? null
  const hub = state?.hub ?? null
  const activity = hub && hub.mode !== 'menu' ? hub.mode : null // learn, read or quiz, when the tutor is menu-driven

  const reading = state?.reading
  let status = 'Looking for the tutor…'
  if (reach === 'no') status = 'I cannot reach the tutor. Start it on the laptop: python tutor_server.py --mode explore --sheet alphabet --paper'
  else if (state && !state.camera.ok) status = 'The camera is not giving pictures. Check that it is connected and not used by another program.'
  else if (state && reading?.between_pages) status = 'Looking for the next page… show it to the camera.'
  else if (state && !state.page.ok) status = 'I cannot see the page. Hold the whole sheet in view of the camera.'
  else if (state && reading && reading.total > 0 && reading.locked === reading.total) {
    status = state.learning ? `The sheet is in view and all ${reading.total} cells are read.` : `All ${reading.total} cells read. Rest a finger on a cell to hear it.`
  }
  else if (state && reading) status = `Reading the sheet… ${reading.locked} of ${reading.total} cells locked in.`
  else if (state) status = 'Page found.'

  const onVideoClick = (e: MouseEvent<HTMLImageElement>) => {
    const r = e.currentTarget.getBoundingClientRect() // a click stands in for the fingertip if the camera cannot track it
    sendFinger((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height)
  }

  const finger = state?.finger.cell
  const commands = activity === 'quiz' ? QUIZ_COMMANDS : activity === 'read' ? READ_COMMANDS : learning ? LEARN_COMMANDS : COMMANDS
  const buttons = commands.filter((c) => state?.config.commands.includes(c))
  const voiceProblems = state?.config.voice && !state.config.voice.ok ? state.config.voice.problems : []
  const said = state ? [...state.said].reverse().slice(0, 5) : []

  if (hub && hub.mode === 'menu') return <Navigate to="/modes" replace /> // nothing chosen yet (or they went back): choose
  if (hub && !loading && !user) return <Navigate to="/login" replace />

  return (
    <main style={styles.container}>
      <div style={{ ...styles.card, maxWidth: 980, alignItems: 'stretch' }}>
        <h1 style={styles.title}>{activity ? TITLES[activity] : learning ? 'Learn braille' : 'Practice'}</h1>
        <p style={styles.subtitle} role="status" aria-live="polite">
          {status}
        </p>

        {learning && <LearnPanel learning={learning} progress={state?.progress} />}
        {activity === 'quiz' && state && <QuizPanel tutor={state.tutor} />}
        {activity === 'read' && <ReadPanel said={said.map((s) => s.text)} />}

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
            <CustomButton key={c} buttonText={LABELS[c] ?? c.charAt(0).toUpperCase() + c.slice(1)} onClick={() => sendCommand(c)} />
          ))}
        </div>

        {learning && user?.kind === 'guest' && (
          <p role="status" style={{ margin: 0, fontSize: 13, color: themeTokens.colors.textMuted }}>
            You are using Braillie as a guest, so your progress is not saved.
          </p>
        )}
        {learning && user?.kind === 'google' && sync.detail && (
          <p role="status" aria-live="polite" style={{ margin: 0, fontSize: 13, color: sync.status === 'error' ? themeTokens.colors.badgeText : themeTokens.colors.textMuted }}>
            {sync.detail}
          </p>
        )}

        {hub && (
          <CustomButton
            buttonText="Change what I am doing"
            onClick={() => {
              void setMode('menu').then(() => navigate('/modes'))
            }}
          />
        )}

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
