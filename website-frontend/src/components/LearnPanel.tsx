import { themeTokens } from '../styles-react-components/mainStyles.tsx'
import CustomButton from '../styles-react-components/CustomButton.tsx'
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
const c = themeTokens.colors

const card = { padding: 16, borderRadius: 16, border: `1px solid ${c.cardBorder}`, background: 'rgba(15, 23, 42, 0.5)', textAlign: 'left' as const }
const heading = { fontSize: 13, textTransform: 'uppercase' as const, letterSpacing: '0.06em', color: c.textMuted, margin: '0 0 8px' }

/** A tile's colour: the more solid the letter, the fuller the purple. */
const tileColour = (m: number) => `rgba(168, 85, 247, ${0.12 + 0.75 * m})`

export function MasteryMap({ progress, current }: { progress: ProgressSummary | null | undefined; current?: string | null }) {
  return (
    <div role="group" aria-label="How well each letter is known" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(38px, 1fr))', gap: 6 }}>
      {ALPHABET.map((l) => {
        const m = progress?.mastery[l] ?? 0
        const practised = l in (progress?.mastery ?? {})
        return (
          <div
            key={l}
            aria-label={`Letter ${l.toUpperCase()}: ${practised ? `${Math.round(m * 100)} percent known` : 'not practised yet'}`}
            style={{
              height: 38, display: 'grid', placeItems: 'center', borderRadius: 10, fontWeight: 700, fontSize: 16,
              background: practised ? tileColour(m) : 'rgba(148, 163, 184, 0.1)',
              color: practised && m > 0.4 ? '#fff' : c.textSecondary,
              outline: current === l ? `2px solid ${c.textPrimary}` : 'none',
            }}
          >
            {l.toUpperCase()}
          </div>
        )
      })}
    </div>
  )
}

/** The learning screen: where you are in the lessons, the letter being learned, what is known so far. The tutor does the speaking. */
const LearnPanel = ({ learning, progress }: { learning: LearningStatus; progress: ProgressSummary | null | undefined }) => {
  const { phase, lesson, target, target_dots: dots } = learning
  const asking = (phase === 'teach' || phase === 'practice' || phase === 'review') && !!target
  const done = Math.max(0, learning.letters - learning.remaining)
  const showDots = phase === 'teach' || learning.hints > 0 || learning.tries > 0 // in practice the pattern is a hint, not a giveaway
  const lessonScore = lesson && progress?.lessons[lesson.id]
  const confusion = progress?.confusions[0]

  return (
    <section aria-label="Your lesson" style={{ display: 'grid', gap: 16 }}>
      <div style={{ ...card, display: 'grid', gap: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap', alignItems: 'baseline' }}>
          <div>
            <p style={heading}>{lesson ? `Lesson ${lesson.index} of ${lesson.of}` : 'Guided lessons'}</p>
            <h2 style={{ margin: 0, fontSize: 22 }}>{lesson ? lesson.title : 'Braille, one letter at a time'}</h2>
          </div>
          <span style={{ padding: '4px 12px', borderRadius: 999, background: c.badgeBg, border: `1px solid ${c.badgeBorder}`, color: c.badgeText, fontSize: 14, fontWeight: 600 }}>
            {PHASE[phase]}
          </span>
        </div>

        {asking && target && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }} aria-live="polite">
            <div style={{ fontSize: 104, fontWeight: 800, lineHeight: 1, minWidth: 90, textAlign: 'center' }} aria-label={`The letter ${target.toUpperCase()}`}>
              {target.toUpperCase()}
            </div>
            {showDots ? (
              <>
                <DotCell dots={dots} size={124} />
                <p style={{ margin: 0, color: c.textSecondary, maxWidth: 260 }}>{describeDots(dots)}</p>
              </>
            ) : (
              <p style={{ margin: 0, color: c.textMuted }}>Find it by touch. Say “hint” if you would like help.</p>
            )}
          </div>
        )}

        {(phase === 'teach' || phase === 'practice') && learning.letters > 0 && (
          <div>
            <div role="progressbar" aria-valuemin={0} aria-valuemax={learning.letters} aria-valuenow={done} aria-label="Progress through this part of the lesson"
              style={{ height: 10, borderRadius: 999, background: 'rgba(148, 163, 184, 0.2)', overflow: 'hidden' }}>
              <div style={{ width: `${(100 * done) / learning.letters}%`, height: '100%', background: c.accent, transition: 'width 0.3s' }} />
            </div>
            <p style={{ margin: '6px 0 0', fontSize: 13, color: c.textMuted }}>
              {phase === 'teach' ? 'Meeting the letters' : learning.round > 1 ? 'Going over the tricky ones' : 'Practising'}: {done} of {learning.letters}
              {learning.in_a_row >= 3 && ` · ${learning.in_a_row} in a row`}
            </p>
          </div>
        )}

        {phase === 'recap' && lessonScore && (
          <p style={{ margin: 0, color: c.textSecondary }}>
            You found {Math.round(lessonScore.last * 100)}% of this lesson’s letters first time{lessonScore.best >= 0.8 ? ' — lesson complete.' : '. It will come round again.'}
          </p>
        )}

        {(phase === 'idle' || phase === 'done') && (
          <div>
            <CustomButton buttonText={phase === 'done' ? 'Start again' : (progress?.sessions ?? 0) > 0 ? 'Continue learning' : 'Start learning'} onClick={() => sendCommand('start quiz')} />
          </div>
        )}
      </div>

      <div style={card}>
        <p style={heading}>Your alphabet</p>
        <MasteryMap progress={progress} current={target} />
        <p style={{ margin: '10px 0 0', fontSize: 13, color: c.textMuted }}>
          Fuller purple = better known. {progress ? `${progress.learned} of 26 known well, ${progress.practised} practised.` : ''}
        </p>
      </div>

      {progress && (
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            [String(progress.streak), progress.streak === 1 ? 'day in a row' : 'days in a row'],
            [String(progress.sessions), progress.sessions === 1 ? 'session' : 'sessions'],
            [String(progress.practised), progress.practised === 1 ? 'letter practised' : 'letters practised'],
          ].map(([n, label]) => (
            <div key={label} style={{ ...card, flex: '1 1 120px', textAlign: 'center' }}>
              <div style={{ fontSize: 30, fontWeight: 800 }}>{n}</div>
              <div style={{ fontSize: 13, color: c.textMuted }}>{label}</div>
            </div>
          ))}
        </div>
      )}

      {confusion && confusion.count >= 2 && (
        <p style={{ ...card, margin: 0, color: c.textSecondary }}>
          Worth watching: <b>{confusion.touched.toUpperCase()}</b> and <b>{confusion.wanted.toUpperCase()}</b> keep getting mixed up. Practice will focus on them.
        </p>
      )}
    </section>
  )
}

export default LearnPanel
