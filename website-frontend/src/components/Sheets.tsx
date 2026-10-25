import type { ReactNode } from 'react'

// The practice sheets, laid out like "Meet Braillo": on one side what to do (the heading, the three steps to make a sheet, each with its
// picture, and one key that downloads all three, an A4 PDF of the "poke side", a page each); on the other, the three pages, which fan
// out like paper on a desk as they scroll into view, the one you point at lifting forward.

const PAGES = ['alphabet', 'words', 'lookalikes']

const STEPS: { label: string; text: string; icon: ReactNode }[] = [
  {
    label: 'Print',
    text: 'At 100% on A4 paper.',
    icon: (
      <>
        <path d="M7 9V4h10v5" />
        <rect x="3" y="9" width="18" height="8" rx="2" />
        <path d="M7 14h10v6H7z" />
      </>
    ),
  },
  {
    label: 'Press',
    text: 'Face down on something soft, push every dot through.',
    icon: <path d="M15 3l6 6-9.5 9.5-3.5.5.5-3.5zM4 20h6" />,
  },
  {
    label: 'Feel',
    text: 'Turn it over: the bumps are your braille.',
    icon: <path d="M8 13V5.5a1.5 1.5 0 0 1 3 0V11m0-1.5a1.5 1.5 0 0 1 3 0V11m0-1a1.5 1.5 0 0 1 3 0v4.5c0 3.6-2.4 6.5-6 6.5-2.6 0-4.2-1.3-5.4-3.4L4.2 15a1.5 1.5 0 0 1 2.6-1.5L8 15" />,
  },
]

const Sheets = () => (
  <div className="sheets">
    <div className="sheets__copy">
      <h2 id="sheets-h">Print your sheets</h2>
      <p>Three pages: the alphabet, words, and letters that look alike.</p>
      <ul className="traits sheets__steps" aria-label="How to make a sheet">
        {STEPS.map((s) => (
          <li key={s.label} className="trait">
            <span className="trait__icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" focusable="false">
                {s.icon}
              </svg>
            </span>
            <span>
              <b>{s.label}</b>
              {s.text}
            </span>
          </li>
        ))}
      </ul>
      <a className="cta sheets__get" href="/sheets/braillie-sheets.pdf" download>
        <svg className="btn__icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <path d="M12 4v11m-5-5 5 5 5-5M5 20h14" />
        </svg>
        Download all sheets<span className="sr-only">: the alphabet, words and look-alikes sheets, one PDF to print on A4</span>
      </a>
    </div>
    <div className="sheets__fan" aria-hidden="true">
      {PAGES.map((p) => (
        <img key={p} className={`sheets__page sheets__page--${p}`} src={`/sheets/page-${p}.png`} alt="" width={520} height={735} loading="lazy" />
      ))}
    </div>
  </div>
)

export default Sheets
