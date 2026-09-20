import { useEffect, type MouseEvent } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import BigButton from '../components/BigButton.tsx'
import { TUTOR_API, clearFinger, sendCommand, sendFinger, setMode } from '../tutor/tutorApi.ts'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useProgressSync } from '../tutor/useProgressSync.ts'
import { useTutorSession } from '../tutor/useTutorSession.ts'
import { useUser } from '../auth/useUser.ts'
import LearnPanel, { AlphabetCard } from '../components/LearnPanel.tsx'
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

  // What a screen reader is told: only what changes the situation (no camera, sheet lost or found, all read), never the running count of cells
  // that would otherwise interrupt every second. The full text stays on screen.
  const announce =
    reach === 'no' || (state && (!state.camera.ok || reading?.between_pages || !state.page.ok))
      ? status
      : state && reading && reading.total > 0 && reading.locked === reading.total
        ? status
        : state
          ? 'The sheet is in view.'
          : ''

  const pageTitle = activity ? TITLES[activity] : learning ? 'Learn braille' : 'Practice'
  useEffect(() => {
    document.title = `${pageTitle} – Braillie` // (the activity is only known once the tutor has answered)
  }, [pageTitle])

  const onVideoClick = (e: MouseEvent<HTMLImageElement>) => {
    const r = e.currentTarget.getBoundingClientRect() // a click stands in for the fingertip if the camera cannot track it
    sendFinger((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height)
  }

  const finger = state?.finger.cell
  const posted = state?.finger.source === 'posted' // a click is standing in for the camera: say so, and offer it back
  const commands = activity === 'quiz' ? QUIZ_COMMANDS : activity === 'read' ? READ_COMMANDS : learning ? LEARN_COMMANDS : COMMANDS
  const buttons = commands.filter((c) => state?.config.commands.includes(c))
  const voiceProblems = state?.config.voice && !state.config.voice.ok ? state.config.voice.problems : []
  const said = state ? [...state.said].reverse().slice(0, 2) : []

  if (hub && hub.mode === 'menu') return <Navigate to="/modes" replace /> // nothing chosen yet (or they went back): choose
  if (!loading && !user) return <Navigate to="/login" replace /> // the way in is the sign-in page, whatever the address

  const sheet = state?.config.sheet ? SHEET_NAMES[state.config.sheet] ?? state.config.sheet : null

  const log =
    said.length > 0 ? (

        <section className="practice__log" aria-labelledby="said-h">
          <h2 id="said-h" className="label">
            What Braillie said
          </h2>
          <ul className="log">
            {said.map((s) => (
              <li key={s.t}>{s.text}</li>
            ))}
          </ul>
        </section>
    ) : null

  const controls = (
        <div className="practice__controls">
          <h2 className="sr-only">Controls</h2>
          <div className="actions">
            {buttons.map((c, i) => (
              <BigButton key={c} variant={i === 0 && learning ? 'primary' : 'secondary'} onClick={() => sendCommand(c)}>
                {LABELS[c] ?? c.charAt(0).toUpperCase() + c.slice(1)}
              </BigButton>
            ))}
          </div>
          {learning && user?.kind === 'guest' && (
            <p className="hint small" role="status">
              You are using Braillie as a guest, so your progress is not saved.
            </p>
          )}
          {learning && user?.kind === 'google' && sync.detail && (
            <p className={sync.status === 'error' ? 'note note--error' : 'hint small'} role="status" aria-live="polite">
              {sync.detail}
            </p>
          )}
          {hub && (
            <BigButton
              variant="secondary"
              onClick={() => {
                void setMode('menu').then(() => navigate('/modes'))
              }}
            >
              Change what I am doing
            </BigButton>
          )}
        </div>
  )

  return (
    <div className="practice">
      <header className="practice__head">
        <p className="eyebrow">{sheet ? `Sheet: ${sheet}` : 'Practice'}</p>
        <h1>{pageTitle}</h1>
        <p className="status">{status}</p>
        <p className="sr-only" role="status" aria-live="polite">
          {announce}
        </p>
        {voiceProblems.length > 0 && (
          <div className="note note--error" role="alert">
            <b>You will not hear the tutor.</b>
            <ul>
              {voiceProblems.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          </div>
        )}
      </header>

      <div className="practice__side">
        {learning ? (
          <LearnPanel learning={learning} progress={state?.progress}>
            {controls}
            {log}
          </LearnPanel>
        ) : (
          <>
            <div className="practice__activity">
              {activity === 'quiz' && state && <QuizPanel tutor={state.tutor} />}
              {activity === 'read' && <ReadPanel said={said.map((s) => s.text)} />}
            </div>
            {controls}
            {log}
          </>
        )}
      </div>

      <div className="practice__left">
        <section className="practice__camera" aria-label="Camera">
          <div className="live">
            <img
              src={`${TUTOR_API}/api/video`}
              alt="Live camera view. Each braille cell the tutor detects has a box around it, with the dots it sees in red and the letter it reads above."
              onClick={onVideoClick}
              style={{ cursor: 'crosshair' }}
            />
          </div>
          <p className="hint small">
            <span className="badge" data-ok={!!state?.page.ok}>
              {state?.page.ok ? 'Sheet in view' : 'Sheet not in view'}
            </span>{' '}
            Green box: read and locked in. Amber: still reading. If the camera loses your fingertip, click the picture where it is.
          </p>
          {posted && (
            <p className="hint small" role="status">
              Braillie is taking the spot you clicked as your fingertip for a few seconds, not what the camera sees.{' '}
              <button type="button" className="btn btn--quiet" onClick={() => clearFinger()}>
                Use the camera instead
              </button>
            </p>
          )}
          <dl className="readout">
          <div>
            <dt>Sheet</dt>
            <dd>{sheet ?? '–'}</dd>
          </div>
          <div>
            <dt>Finger</dt>
            <dd>{finger ? `${finger.label ?? finger.letter ?? 'a cell'}: dots ${finger.dots.join(', ')}` : 'not on a cell'}</dd>
          </div>
          <div>
            <dt>Tutor</dt>
            <dd>{state?.tutor.state ?? '–'}</dd>
          </div>
        </dl>
        </section>

        {learning && <AlphabetCard learning={learning} progress={state?.progress} />}
      </div>
    </div>
  )
}

export default Practice
