import { useEffect, useState, type MouseEvent, type ReactNode } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import DotCell from '../components/DotCell.tsx'
import Lyrics from '../components/Lyrics.tsx'
import Mascot from '../components/Mascot.tsx'
import MicButton from '../components/MicButton.tsx'
import { describeDots } from '../tutor/describeDots.ts'
import { TUTOR_API, sendCommand, sendFinger, setMic, setMode, type LearningStatus, type TutorState } from '../tutor/tutorApi.ts'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useProgressSync } from '../tutor/useProgressSync.ts'
import { useTutorSession } from '../tutor/useTutorSession.ts'
import { useSpeechPosition } from '../voice/useSpeechPosition.ts'
import { useUser } from '../auth/useUser.ts'

const TITLES: Record<string, string> = { learn: 'Learn', read: 'Read', quiz: 'Quiz' }

// Line pictures for the keys (not characters, which some fonts lack)
const ICONS: Record<string, ReactNode> = {
  hint: <path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0 0 12 3z" />,
  found: <path d="M4.5 12.5l4.5 4.5 10.5-11" />,
  next: <path d="M4 12h16m-6-6 6 6-6 6" />,
  stop: <path d="M5 21V4m0 0h11l-2 4 2 4H5" />,
  start: <path d="M8 5v14l11-7z" />,
  explore: <path d="M8 13V5.5a1.5 1.5 0 0 1 3 0V12m0-1.5v-2a1.5 1.5 0 0 1 3 0V12m0-1a1.5 1.5 0 0 1 3 0v4a6 6 0 0 1-6 6h-1a6 6 0 0 1-5-2.7L3.5 15a1.5 1.5 0 0 1 2.5-1.7L8 15" />,
  repeat: <path d="M4 12a8 8 0 1 0 2.5-5.8M4 4v4.5h4.5" />,
  home: <path d="M4 11.5 12 4l8 7.5M6.5 9.5V20h11V9.5" />,
  camera: (
    <>
      <rect x="3" y="6.5" width="18" height="13" rx="3" />
      <circle cx="12" cy="13" r="3.5" />
      <path d="M8.5 6.5 10 4h4l1.5 2.5" />
    </>
  ),
}
const Icon = ({ name }: { name: string }) => (
  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    {ICONS[name]}
  </svg>
)

/** One thing to do, as a big round picture with a word under it. The first one of a row is filled in: the usual next step. */
const Key = ({ icon, label, command, main = false }: { icon: string; label: string; command: string; main?: boolean }) => (
  <button type="button" className={main ? 'key key--main' : 'key'} onClick={() => void sendCommand(command)}>
    <span className="key__disc">
      <Icon name={icon} />
    </span>
    <span className="key__label">{label}</span>
  </button>
)

/** How far along: a dot per letter (or question), filled once done. */
const Dots = ({ done, of, label }: { done: number; of: number; label: string }) => (
  <div className="steps" role="img" aria-label={label}>
    {Array.from({ length: of }, (_, i) => (
      <i key={i} data-on={i < done} data-now={i === done} />
    ))}
  </div>
)

/** The letter to find, big, and (when it should be shown) its dots beside it. */
const Target = ({ letter, dots, showDots }: { letter: string; dots: number[]; showDots: boolean }) => (
  <div className="target" aria-live="polite">
    <div className="target__letter" key={letter} aria-label={`The letter ${letter.toUpperCase()}`}>
      {letter.toUpperCase()}
    </div>
    {showDots && (
      <div className="target__dots">
        <DotCell dots={dots} size={150} tone="accent" label={`${letter.toUpperCase()}: ${describeDots(dots)}`} />
      </div>
    )}
  </div>
)

/** Learn: the lesson, the letter being learned, how far through, and the few keys that matter now. */
const LearnStage = ({ learning, state }: { learning: LearningStatus; state: TutorState }) => {
  const { phase, lesson, target, target_dots: dots } = learning
  const asking = (phase === 'warmup' || phase === 'teach' || phase === 'practice' || phase === 'review') && !!target
  const done = Math.max(0, learning.letters - learning.remaining)
  const showDots = phase === 'teach' || phase === 'warmup' || learning.hints > 0 || learning.tries > 0 // in practice the pattern is a hint, not a giveaway
  const score = lesson && state.progress?.lessons[lesson.id]
  const finger = state.finger.cell
  return (
    <>
      {lesson && (
        <p className="play__kicker">{phase === 'warmup' ? 'Going over letters from before' : `Lesson ${lesson.index} of ${lesson.of} · ${lesson.title}`}</p>
      )}
      {asking && target ? (
        <>
          <Target letter={target} dots={dots} showDots={showDots} />
          {learning.letters > 0 && phase !== 'warmup' && <Dots done={done} of={learning.letters} label={`${done} of ${learning.letters} letters done`} />}
          <div className="keys">
            <Key icon="found" label="Found it" command="found it" main />
            <Key icon="hint" label="Hint" command="hint" />
            <Key icon="next" label="Next" command="next" />
            <Key icon="stop" label="Finish" command="stop" />
          </div>
        </>
      ) : phase === 'explore' ? (
        <>
          <p className="play__big" aria-live="polite">
            {finger?.letter ? finger.letter.toUpperCase() : 'Touch a letter'}
          </p>
          <div className="keys">
            <Key icon="start" label="Practise" command="practice" main />
            <Key icon="stop" label="Finish" command="stop" />
          </div>
        </>
      ) : (
        <>
          <p className="play__big play__big--words">
            {phase === 'recap' && score
              ? `${Math.round(score.last * 100)}% first time`
              : phase === 'await_sheet'
                ? 'Put the alphabet sheet down'
                : phase === 'done'
                  ? 'Well done!'
                  : 'Ready?'}
          </p>
          <div className="keys">
            <Key icon="start" label={phase === 'done' ? 'Again' : phase === 'recap' ? 'Next' : 'Start'} command="start quiz" main />
            <Key icon="explore" label="Explore" command="explore" />
          </div>
        </>
      )}
    </>
  )
}

/** Quiz: which question, the letter to find, how many are right. */
const QuizStage = ({ tutor }: { tutor: TutorState['tutor'] }) => {
  const asking = tutor.state === 'asking'
  const total = tutor.total ?? 0
  const letter = /letter ([A-Za-z])\b/.exec(tutor.prompt)?.[1]
  const asked = tutor.asked ?? 0
  const right = tutor.correct ?? 0
  return (
    <>
      <p className="play__kicker">{asking ? `Question ${tutor.question} of ${total}` : tutor.state === 'done' ? 'Quiz finished' : 'Getting ready'}</p>
      {asking && letter ? (
        <>
          <Target letter={letter} dots={[]} showDots={false} />
          {total > 0 && <Dots done={asked} of={total} label={`${asked} answered, ${right} right`} />}
          <p className="play__score">
            {right} of {asked} right
          </p>
          <div className="keys">
            <Key icon="found" label="Found it" command="found it" main />
            <Key icon="hint" label="Hint" command="hint" />
            <Key icon="next" label="Next" command="next" />
            <Key icon="stop" label="Finish" command="stop" />
          </div>
        </>
      ) : (
        <p className="play__big play__big--words">{tutor.state === 'done' ? `${right} of ${asked} right` : 'Ready?'}</p>
      )}
    </>
  )
}

/** Read: the last word read out, big. */
const ReadStage = ({ said }: { said: string[] }) => {
  const last = said.find((t) => t.startsWith('The word is') || t.startsWith('I think it says') || t.startsWith("I couldn't read"))
  const word = last ? /^(?:The word is|I think it says)\s+(.+?)\.?$/.exec(last)?.[1] : null
  return (
    <>
      <p className="play__kicker">Words sheet</p>
      <p className="play__big" aria-live="polite">
        {word ?? (last ? '?' : 'Touch a word')}
      </p>
      <div className="keys">
        <Key icon="stop" label="Finish" command="stop" />
      </div>
    </>
  )
}

/**
 * Learn, Read and Quiz: laid out like home. Braillo on the left saying what to do (his words as captions while he speaks), with round
 * keys under him (mute, say it again, the camera, home); on the right only what matters now: the letter or word, how far along, and the
 * few things to do as big round pictures. A short warning appears only when something is wrong (no camera, the sheet out of view).
 */
const Practice = () => {
  const { state, reach } = useTutorState(500)
  const navigate = useNavigate()
  const { user, loading } = useUser()
  useTutorSession(user, state)
  const sync = useProgressSync(state, user?.kind === 'google')
  const [camera, setCamera] = useState(false)
  const learning = state?.learning ?? null
  const hub = state?.hub ?? null
  const activity = hub && hub.mode !== 'menu' ? hub.mode : learning ? 'learn' : null
  const reading = state?.reading

  // only what is wrong, in a few words (nothing while all is well)
  let problem = ''
  if (reach === 'no') problem = 'I cannot reach the tutor. Start it on the laptop.'
  else if (state && !state.camera.ok) problem = 'The camera is not giving pictures.'
  else if (state && reading?.between_pages) problem = 'Looking for the next page. Show it to the camera.'
  else if (state && !state.page.ok) problem = 'I cannot see the sheet. Hold all of it in view.'
  const voiceProblems = state?.config.voice && !state.config.voice.ok ? state.config.voice.problems : []

  const title = activity ? TITLES[activity] : 'Practice'
  useEffect(() => {
    document.title = `${title} – Braillie` // (the activity is only known once the tutor has answered)
  }, [title])

  const speaking = !!state?.speaking && !!state?.speech
  const spoken = useSpeechPosition(speaking ? state?.speech : null, state?.settings?.speech_speed)
  const listening = !speaking && !!state?.heard
  const muted = !!state?.muted
  const lastSaid = state?.said.length ? state.said[state.said.length - 1].text : ''

  if (hub && hub.mode === 'menu') return <Navigate to="/modes" replace /> // nothing chosen yet (or they went back): choose
  if (!loading && !user) return <Navigate to="/login" replace /> // the way in is the sign-in page, whatever the address

  const onVideoClick = (e: MouseEvent<HTMLImageElement>) => {
    const r = e.currentTarget.getBoundingClientRect() // a click stands in for the fingertip if the camera cannot track it
    void sendFinger((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height)
  }
  const home = () => void setMode('menu').then(() => navigate('/modes'))

  return (
    <div className="home2 play">
      <h1 className="sr-only">{title}</h1>
      <p className="sr-only" role="status" aria-live="polite">
        {problem}
      </p>

      <section className="home2__braillo voicecard" aria-label="Braillo, your tutor">
        <div className="voicecard__stage">
          <Mascot state={speaking ? 'speaking' : listening ? 'listening' : 'idle'} words={spoken.sentence * 1000 + spoken.said} />
          <div className="voicecard__words">
            <div className="voicecard__body">
              {speaking && state?.speech ? (
                <Lyrics speech={state.speech} speed={state.settings?.speech_speed} />
              ) : (
                <p className="voicecard__hint" aria-hidden="true">
                  {lastSaid || 'Ready when you are'}
                </p>
              )}
            </div>
          </div>
        </div>
        <div className="voicecard__foot">
          {reach === 'yes' && (
            <MicButton state={muted ? 'muted' : 'live'} label={muted ? 'Turn the microphone on' : 'Mute the microphone'} onClick={() => void setMic(muted).catch(() => {})} />
          )}
          <button type="button" className="roundkey" aria-label="Say that again" title="Say that again" onClick={() => void sendCommand('repeat')}>
            <Icon name="repeat" />
          </button>
          <button type="button" className="roundkey" aria-label="Show the camera" title="Show the camera" aria-pressed={camera} onClick={() => setCamera((c) => !c)}>
            <Icon name="camera" />
          </button>
          {hub && (
            <button type="button" className="roundkey" aria-label="Home: choose something else" title="Home" onClick={home}>
              <Icon name="home" />
            </button>
          )}
        </div>
      </section>

      <section className="play__side" aria-label={title}>
        {problem && <p className="note note--error play__problem">{problem}</p>}
        {voiceProblems.length > 0 && (
          <div className="note note--error" role="alert">
            <b>You will not hear the tutor.</b> {voiceProblems.join(' ')}
          </div>
        )}

        {activity === 'learn' && learning && state && <LearnStage learning={learning} state={state} />}
        {activity === 'quiz' && state && <QuizStage tutor={state.tutor} />}
        {activity === 'read' && state && <ReadStage said={[...state.said].reverse().slice(0, 3).map((s) => s.text)} />}

        {camera && (
          <div className="live play__camera">
            <img
              src={`${TUTOR_API}/api/video`}
              alt="Live camera view. Each braille cell the tutor detects has a box around it, with the letter it reads above. Click where your finger is if the tutor cannot see it."
              onClick={onVideoClick}
              style={{ cursor: 'crosshair' }}
            />
          </div>
        )}

        {learning && user?.kind === 'google' && sync.status === 'error' && sync.detail && (
          <p className="note note--error" role="status">
            {sync.detail}
          </p>
        )}
      </section>
    </div>
  )
}

export default Practice
