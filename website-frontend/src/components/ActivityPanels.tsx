import { themeTokens } from '../styles-react-components/mainStyles.tsx'
import type { TutorState } from '../tutor/tutorApi.ts'

const c = themeTokens.colors
const card = { padding: 16, borderRadius: 16, border: `1px solid ${c.cardBorder}`, background: 'rgba(15, 23, 42, 0.5)', textAlign: 'left' as const }

/** The quiz: which question, the letter being asked for, and the score so far. The tutor asks out loud; this is the same on screen. */
export function QuizPanel({ tutor }: { tutor: TutorState['tutor'] }) {
  const asking = tutor.state === 'asking'
  const total = tutor.total ?? 0
  const letter = /letter ([A-Za-z])\b/.exec(tutor.prompt)?.[1]
  return (
    <section aria-label="Your quiz" style={{ ...card, display: 'grid', gap: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, alignItems: 'baseline' }}>
        <h2 style={{ margin: 0, fontSize: 22 }}>Look-alike letters quiz</h2>
        <span style={{ color: c.textMuted }}>{asking ? `Question ${tutor.question} of ${total}` : tutor.state === 'done' ? 'Finished' : 'Getting ready'}</span>
      </div>
      {asking && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }} aria-live="polite">
          {letter && (
            <div style={{ fontSize: 104, fontWeight: 800, lineHeight: 1, minWidth: 90, textAlign: 'center' }} aria-label={`The letter ${letter.toUpperCase()}`}>
              {letter.toUpperCase()}
            </div>
          )}
          <p style={{ margin: 0, color: c.textSecondary, fontSize: 18 }}>{tutor.prompt} Rest your finger on it.</p>
        </div>
      )}
      <p style={{ margin: 0, color: c.textMuted }}>
        {tutor.asked ?? 0} answered · {tutor.correct ?? 0} right{(tutor.tries ?? 0) > 0 ? ` · ${tutor.tries} miss${tutor.tries === 1 ? '' : 'es'} on this one` : ''}
      </p>
    </section>
  )
}

/** Read mode: what to do, and the last thing the tutor read out. */
export function ReadPanel({ said }: { said: string[] }) {
  const last = said.find((t) => t.startsWith('The word is') || t.startsWith('I think it says') || t.startsWith("I couldn't read"))
  return (
    <section aria-label="Reading" style={{ ...card, display: 'grid', gap: 10 }}>
      <h2 style={{ margin: 0, fontSize: 22 }}>Reading words</h2>
      <p style={{ margin: 0, color: c.textSecondary, fontSize: 18 }}>Rest a finger on a word on the words sheet and the tutor reads it aloud.</p>
      {last && (
        <p style={{ margin: 0, fontSize: 28, fontWeight: 700 }} aria-live="polite">
          {last}
        </p>
      )}
    </section>
  )
}
