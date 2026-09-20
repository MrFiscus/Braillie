import type { TutorState } from '../tutor/tutorApi.ts'

/** The quiz: which question, the letter being asked for, and the score so far. The tutor asks out loud; this is the same on screen. */
export function QuizPanel({ tutor }: { tutor: TutorState['tutor'] }) {
  const asking = tutor.state === 'asking'
  const total = tutor.total ?? 0
  const letter = /letter ([A-Za-z])\b/.exec(tutor.prompt)?.[1]
  return (
    <section className="card card--shadow" aria-label="Your quiz">
      <div className="card__top">
        <h2>Look-alike letters</h2>
        <span className="badge">{asking ? `Question ${tutor.question} of ${total}` : tutor.state === 'done' ? 'Finished' : 'Getting ready'}</span>
      </div>
      {asking && (
        <div className="ask" aria-live="polite">
          {letter && (
            <div className="bigletter" aria-label={`The letter ${letter.toUpperCase()}`}>
              {letter.toUpperCase()}
            </div>
          )}
          <p className="hint">{tutor.prompt} Rest your finger on it, or click the picture where it is.</p>
        </div>
      )}
      <p className="hint">
        {tutor.asked ?? 0} answered · {tutor.correct ?? 0} right{(tutor.tries ?? 0) > 0 ? ` · ${tutor.tries} miss${tutor.tries === 1 ? '' : 'es'} on this one` : ''}
      </p>
    </section>
  )
}

/** Read mode: what to do, and the last thing the tutor read out. */
export function ReadPanel({ said }: { said: string[] }) {
  const last = said.find((t) => t.startsWith('The word is') || t.startsWith('I think it says') || t.startsWith("I couldn't read"))
  return (
    <section className="card card--shadow" aria-label="Reading">
      <h2>Reading words</h2>
      <p className="hint">Rest a finger on a word on the words sheet and Braillie reads it aloud.</p>
      {last && (
        <p className="bigword" aria-live="polite">
          {last}
        </p>
      )}
    </section>
  )
}
