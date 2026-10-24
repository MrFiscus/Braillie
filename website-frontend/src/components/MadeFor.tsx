import type { ReactNode } from 'react'

// Who Braillie is made for, as four pictures with a few words each (the longer sentence is there for a screen reader). Pointing at
// one plays its picture: the microphone bobs, the screen reader's sound beams out, the eye blinks and looks round, the heart beats.

const ITEMS: { kind: string; title: string; short: string; full: string; icon: ReactNode }[] = [
  {
    kind: 'talk',
    title: 'Talk, don’t type',
    short: 'Say “learn”, “hint”, or ask Braillo anything.',
    full: 'Talk instead of type: say what you want, like learn, hint or repeat, or ask Braillo any question.',
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
    short: 'Every voice action has a key too.',
    full: 'Works with your screen reader: real headings, labels and landmarks, and every voice action has a key or button too.',
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
    short: 'Big type, strong colours, four text sizes.',
    full: 'Easy to see: large, clear type and strong colours, with black-on-white and yellow-on-black themes and four text sizes.',
    icon: (
      <>
        <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" />
        <circle className="madefor__pupil" cx="12" cy="12" r="3.2" />
      </>
    ),
  },
  {
    kind: 'patient',
    title: 'Patient',
    short: 'Brings back the letters you find hard.',
    full: 'Patient and personal: it remembers the letters you find hard and brings them back, and never says only wrong.',
    icon: <path d="M12 20s-7.5-4.6-7.5-10A4.3 4.3 0 0 1 12 7.5 4.3 4.3 0 0 1 19.5 10c0 5.4-7.5 10-7.5 10z" />,
  },
]

const MadeFor = () => (
  <ul className="madefor">
    {ITEMS.map((it) => (
      <li key={it.title} className={`madefor__item madefor__item--${it.kind}`}>
        <span className="madefor__icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" focusable="false">
            {it.icon}
          </svg>
        </span>
        <h3>{it.title}</h3>
        <p aria-hidden="true">{it.short}</p>
        <p className="sr-only">{it.full}</p>
      </li>
    ))}
  </ul>
)

export default MadeFor
