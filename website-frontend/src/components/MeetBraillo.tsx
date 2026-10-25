import { useEffect, useState, type ReactNode } from 'react'
import Mascot, { type MascotState } from './Mascot.tsx'

// "Meet Braillo": the tutor, big, acting out what it does. It talks (its mouth moving), listens (a hand to its ear while you
// speak) and helps, one after another, with what is said in a speech bubble beside it. A picture: a screen reader gets a sentence instead.
// Beside it, the three things it does, one line each, in the same order as the scenes: pointing at one has Braillo act it out
// (and holds it there) until the pointer moves on.

const SCENES: { state: MascotState; who: 'braillo' | 'you'; line: string }[] = [
  { state: 'speaking', who: 'braillo', line: 'The letter D. Dots 1, 4 and 5.' },
  { state: 'listening', who: 'you', line: 'Hint' },
  { state: 'speaking', who: 'braillo', line: 'Feel for the dot top right.' },
]
const SCENE_MS = 3200
const MOUTH_MS = 260 // how often the mouth moves while it talks (a word every quarter second or so)

const TRAITS: { label: string; text: string; icon: ReactNode }[] = [
  { label: 'Talks', text: 'Says each letter and its dots.', icon: <path d="M4 5h16v11H9l-5 4z" /> },
  {
    label: 'Listens',
    text: 'Just say your answer, or “hint”.',
    icon: <path d="M7 15c0 3 2 5 4.5 5S15 18 15 16c0-2 3-3 3-7a6 6 0 0 0-12 0M10 10a2 2 0 0 1 4 0c0 1.5-2 2-2 4" />,
  },
  { label: 'Helps', text: 'Gives a clue when you are stuck.', icon: <path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0 0 12 3z" /> },
]

const MeetBraillo = ({ still }: { still: boolean }) => {
  const [scene, setScene] = useState(0)
  const [held, setHeld] = useState<number | null>(null) // the line being pointed at, if any
  const [mouth, setMouth] = useState(0)
  const shown = held ?? scene
  const s = SCENES[shown]

  useEffect(() => {
    if (still || held !== null) return
    const t = window.setInterval(() => setScene((n) => (n + 1) % SCENES.length), SCENE_MS)
    return () => window.clearInterval(t)
  }, [still, held])

  useEffect(() => {
    if (still || s.state !== 'speaking') return
    const t = window.setInterval(() => setMouth((n) => n + 1), MOUTH_MS)
    return () => window.clearInterval(t)
  }, [still, s.state])

  return (
    <div className="meet">
      <figure
        className="meet__stage"
        role="img"
        aria-label="Braillo, a friendly yellow figure shaped like a braille cell. He tells you about each letter, listens when you answer, and helps when you are stuck."
      >
        <Mascot state={s.state} size={300} words={mouth} />
        <figcaption className={`meet__bubble meet__bubble--${s.who}`} key={shown} aria-hidden="true">
          {s.who === 'you' ? `“${s.line}”` : s.line}
        </figcaption>
      </figure>

      <div className="meet__copy">
        <h2 id="meet-h">Meet Braillo</h2>
        <p className="meet__sub">Your friendly braille tutor.</p>
        <ul className="traits" aria-label="What Braillo does">
          {TRAITS.map((t, i) => (
            <li key={t.label} className={`trait trait--${t.label.toLowerCase()}`} onPointerEnter={() => setHeld(i)} onPointerLeave={() => setHeld(null)}>
              <span className="trait__icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" focusable="false">
                  {t.icon}
                </svg>
              </span>
              <span>
                <b>{t.label}</b>
                {t.text}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

export default MeetBraillo
