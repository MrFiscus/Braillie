import type { ReactNode } from 'react'

// Who Braillie is made for, laid out like the practice sheets: the heading and one line on one side, and
// four cards on the other, each a picture, a title and one sentence. Pointing at a card plays its picture: the microphone bobs, the
// screen reader's sound beams out, the eye blinks and looks round, the heart beats.

const ITEMS: { kind: string; title: string; text: string; icon: ReactNode }[] = [
  {
    kind: 'talk',
    title: 'Voice first',
    text: 'Say “learn”, “hint”, or ask Braillo anything.',
    icon: (
      <>
        <rect x="9" y="3" width="6" height="11" rx="3" />
        <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M8.5 21h7" />
      </>
    ),
  },
  {
    kind: 'reader',
    title: 'Screen reader ready',
    text: 'Every voice action has a key too.',
    icon: (
      <>
        <rect x="2.5" y="4" width="13" height="10" rx="2" />
        <path d="M6 18h6M9 14v4" />
        <path className="madefor__wave" d="M18.5 7.5c1.2 1.3 1.2 3.7 0 5" />
        <path className="madefor__wave" d="M20.8 5.5c2.3 2.5 2.3 6.5 0 9" />
      </>
    ),
  },
  {
    kind: 'see',
    title: 'Easy to see',
    text: 'Big type, strong colours, four text sizes.',
    icon: (
      <>
        <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" />
        <circle className="madefor__pupil" cx="12" cy="12" r="3.2" />
      </>
    ),
  },
  {
    kind: 'patient',
    title: 'Patient by design',
    text: 'Brings back the letters you find hard.',
    icon: <path d="M12 20s-7.5-4.6-7.5-10A4.3 4.3 0 0 1 12 7.5 4.3 4.3 0 0 1 19.5 10c0 5.4-7.5 10-7.5 10z" />,
  },
]

const MadeFor = () => (
  <div className="made">
    <div className="made__copy">
      <h2 id="made-h">
        Made for blind and <span className="made__nowrap">low-vision</span> learners
      </h2>
      <p>Everything works by voice, by keyboard or with a screen reader.</p>
    </div>
    <ul className="madefor">
      {ITEMS.map((it) => (
        <li key={it.title} className={`madefor__item madefor__item--${it.kind}`}>
          <span className="madefor__icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" focusable="false">
              {it.icon}
            </svg>
          </span>
          <h3>{it.title}</h3>
          <p>{it.text}</p>
        </li>
      ))}
    </ul>
  </div>
)

export default MadeFor
