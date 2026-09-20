import type { ReactNode } from 'react'
import BigButton from './BigButton.tsx'
import type { LearningPhase, LearningStatus, ProgressSummary } from '../tutor/tutorApi.ts'
import { sendCommand } from '../tutor/tutorApi.ts'
import DotCell from './DotCell.tsx'
import { describeDots } from '../tutor/describeDots.ts'

const PHASE: Record<LearningPhase, string> = {
  idle: 'Ready',
  await_sheet: 'Waiting for the sheet',
  teach: 'Learning a new letter',
  practice: 'Practising',
  review: 'Review',
  recap: 'Lesson finished',
  explore: 'Free exploring',
  done: 'Session finished',
}
const ALPHABET = 'abcdefghijklmnopqrstuvwxyz'.split('')

/** Every letter, with how well it is known: the number of filled dots under it (0 to 4) as well as the words for a screen reader. */
export function MasteryMap({ progress, current }: { progress: ProgressSummary | null | undefined; current?: string | null }) {
  return (
    <ul className="mastery" aria-label="How well each letter is known">
      {ALPHABET.map((l) => {
        const m = progress?.mastery[l] ?? 0
        const practised = l in (progress?.mastery ?? {})
        const pips = Math.round(m * 4)
        return (
          <li key={l} data-now={current === l} data-none={!practised}>
            <span aria-hidden="true">{l.toUpperCase()}</span>
            <span className="pips" aria-hidden="true">
              {[0, 1, 2, 3].map((i) => (
                <i key={i} data-on={i < pips} />
              ))}
            </span>
            <span className="sr-only">
              Letter {l.toUpperCase()}: {practised ? `${Math.round(m * 100)} percent known` : 'not practised yet'}
              {current === l ? ', the one you are on' : ''}
            </span>
          </li>
        )
      })}
    </ul>
  )
}

/** The learning screen: where you are in the lessons, the letter being learned, what is known so far. The tutor does the speaking. */
const LearnPanel = ({ learning, progress, children }: { learning: LearningStatus; progress: ProgressSummary | null | undefined; children?: ReactNode }) => {
  const { phase, lesson, target, target_dots: dots } = learning
  const asking = (phase === 'teach' || phase === 'practice' || phase === 'review') && !!target
  const done = Math.max(0, learning.letters - learning.remaining)
  const showDots = phase === 'teach' || learning.hints > 0 || learning.tries > 0 // in practice the pattern is a hint, not a giveaway
  const lessonScore = lesson && progress?.lessons[lesson.id]

  return (
    <section aria-label="Your lesson" className="practice__activity">
      <div className="card card--shadow">
        <div className="card__top">
          <div>
            <p className="label">{lesson ? `Lesson ${lesson.index} of ${lesson.of}` : 'Guided lessons'}</p>
            <h2>{lesson ? lesson.title : 'Braille, one letter at a time'}</h2>
          </div>
          <span className="badge">{PHASE[phase]}</span>
        </div>

        {asking && target && (
          <div className="ask" aria-live="polite">
            <div className="bigletter" aria-label={`The letter ${target.toUpperCase()}`}>
              {target.toUpperCase()}
            </div>
            {showDots ? (
              <>
                <DotCell dots={dots} size={104} tone="accent" />
                <p className="hint">{describeDots(dots)}</p>
              </>
            ) : (
              <p className="hint">Find it by touch. Say “hint” if you would like help.</p>
            )}
          </div>
        )}

        {(phase === 'teach' || phase === 'practice') && learning.letters > 0 && (
          <div>
            <div className="meter" role="progressbar" aria-valuemin={0} aria-valuemax={learning.letters} aria-valuenow={done} aria-label="Progress through this part of the lesson">
              <div style={{ width: `${(100 * done) / learning.letters}%` }} />
            </div>
            <p className="hint small">
              {phase === 'teach' ? 'Meeting the letters' : learning.round > 1 ? 'Going over the tricky ones' : 'Practising'}: {done} of {learning.letters}
              {learning.in_a_row >= 3 && ` · ${learning.in_a_row} in a row`}
            </p>
          </div>
        )}

        {phase === 'recap' && lessonScore && (
          <p>
            You found {Math.round(lessonScore.last * 100)}% of this lesson’s letters first time{lessonScore.best >= 0.8 ? ' — lesson complete.' : '. It will come round again.'}
          </p>
        )}

        {(phase === 'idle' || phase === 'done') && (
          <BigButton onClick={() => sendCommand('start quiz')}>
            {phase === 'done' ? 'Start again' : (progress?.sessions ?? 0) > 0 ? 'Continue learning' : 'Start learning'}
          </BigButton>
        )}
      </div>

      {children /* the buttons, right under the lesson: the thing to press should not be below the fold */}

    </section>
  )
}

/** Every letter and how well it is known, with the counts and the pair of letters that keep getting mixed up. */
export const AlphabetCard = ({ learning, progress }: { learning: LearningStatus; progress: ProgressSummary | null | undefined }) => {
  const { target } = learning
  const confusion = progress?.confusions[0]
  return (
    <section aria-label="Your alphabet" className="practice__activity">
      <div className="card">
      <p className="label">Your alphabet</p>
      <MasteryMap progress={progress} current={target} />
      <p className="hint small">
        More filled dots means better known.
        {progress
          ? ` ${progress.learned} of 26 known well, ${progress.practised} practised · ${progress.streak} ${progress.streak === 1 ? 'day' : 'days'} in a row · ${progress.sessions} ${progress.sessions === 1 ? 'session' : 'sessions'}.`
          : ''}
      </p>
    </div>

    {confusion && confusion.count >= 2 && (
      <p className="note">
        <b>Worth watching:</b> {confusion.touched.toUpperCase()} and {confusion.wanted.toUpperCase()} keep getting mixed up. Practice will focus on them.
      </p>
    )}
    </section>
  )
}

export default LearnPanel
