import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import AccountKey from '../components/AccountKey.tsx'
import DisplayMenu from '../components/DisplayMenu.tsx'
import HowItWorks from '../components/HowItWorks.tsx'
import MadeFor from '../components/MadeFor.tsx'
import MeetBraillo from '../components/MeetBraillo.tsx'
import SiteFooter from '../components/SiteFooter.tsx'
import Sheets from '../components/Sheets.tsx'
import { DOTS } from '../braille/alphabet.ts'

const WORD = 'braillie'
const STEP_MS = 1700 // one letter of the word every this long

const arrow = (
  <svg className="arrow" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <path d="M4 12h16m-6-6 6 6-6 6" />
  </svg>
)

/** The picture beside the welcome: a strip of braille spelling "braillie", read the way a learner reads it. A fingertip slides
 * from cell to cell; the cell under it lights up, a bubble says its letter, and the printed word builds up above the strip as each
 * letter is read. A picture only (the words on the page say everything); it keeps still for anyone who has asked their device for
 * less motion. */
const PITCH = 84 // from one cell to the next, in the drawing's units
const FIRST = 52 // the first cell's left edge
const dotAt = (n: number) => ({ x: n <= 3 ? 17 : 51, y: 208 + ((n - 1) % 3) * 34 })

const Reader = ({ at }: { at: number }) => {
  const x = FIRST + at * PITCH + 34 // the middle of the cell being read
  const letter = WORD[at]
  return (
    <svg className="reader" viewBox="0 0 780 480" aria-hidden="true" focusable="false">
      {/* the bubble, following the finger: the letter it is on, said aloud */}
      <g className="reader__follow" style={{ transform: `translateX(${x}px)` }}>
        <g className="reader__bubble" key={at}>
          <rect x="-52" y="0" width="104" height="96" rx="22" />
          <path d="M-14 94 0 114l14-20z" />
          <text x="0" y="72" textAnchor="middle">
            {letter.toUpperCase()}
          </text>
        </g>
        <g className="reader__voice">
          <path d="M66 30c8 9 8 25 0 34" />
          <path d="M82 18c14 16 14 42 0 58" />
        </g>
      </g>

      {/* the printed sheet: the word so far in print along the top, the dots under it */}
      <rect className="reader__sheet" x="20" y="124" width="740" height="214" rx="26" />
      {[...WORD].map((l, n) => {
        const left = FIRST + n * PITCH
        const state = n < at ? 'is-read' : n === at ? 'is-now' : ''
        return (
          <g key={n} className={`reader__cell ${state}`} transform={`translate(${left} 0)`}>
            <rect className="reader__pad" x="-4" y="182" width="76" height="134" rx="16" />
            {[1, 2, 3, 4, 5, 6].map((d) => {
              const { x: cx, y: cy } = dotAt(d)
              const on = DOTS[l]?.includes(d)
              return <circle key={d} className={on ? 'reader__dot is-on' : 'reader__dot'} cx={cx} cy={cy} r={on ? 11 : 4} />
            })}
            <text className="reader__print" x="34" y="166" textAnchor="middle">
              {l}
            </text>
          </g>
        )
      })}

      {/* the fingertip, coming up from below and resting on the cell */}
      <g className="reader__follow" style={{ transform: `translateX(${x}px)` }}>
        <g className="reader__finger">
          <path d="M-30 340c0-26 12-40 30-40s30 14 30 40v150h-60z" />
          <path d="M-16 332c0-12 6-18 16-18s16 6 16 18" className="reader__nail" />
          <path d="M-20 404c12 5 28 5 40 0M-20 426c12 5 28 5 40 0" className="reader__nail" />
        </g>
      </g>
    </svg>
  )
}

/** The front door: what Braillie is, how it works, and who it is made for, with one clear way in. */
const Landing = () => {
  const [i, setI] = useState(0)
  const [lessMotion, setLessMotion] = useState(() => typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const still = lessMotion

  useEffect(() => {
    document.title = 'Braillie – learn braille by touch'
    if (typeof window.matchMedia !== 'function') return
    const q = window.matchMedia('(prefers-reduced-motion: reduce)')
    const change = () => setLessMotion(q.matches)
    q.addEventListener('change', change)
    return () => q.removeEventListener('change', change)
  }, [])

  useEffect(() => {
    if (still) return
    const t = window.setInterval(() => setI((n) => (n + 1) % WORD.length), STEP_MS)
    return () => window.clearInterval(t)
  }, [still])

  const letter = WORD[i]
  const dots = DOTS[letter]

  return (
    <div className={still ? 'landing is-still' : 'landing'} id="top">
      <a className="skip" href="#main">
        Skip to the page
      </a>
      <header className="landing__head">
        <Link className="brand" to="/" aria-label="Braillie, home">
          <span>
            Braillie<span className="brand__dot">.</span>
          </span>
        </Link>
        <div className="landing__nav">
          <AccountKey />
          <DisplayMenu share={false} />
        </div>
      </header>

      <main id="main" tabIndex={-1}>
        <section className="hero" aria-labelledby="hero-h">
          <div className="hero__copy">
            <h1 id="hero-h">
              Feel the dots.
              <br />
              <span>Hear the letters.</span>
            </h1>
            <p className="hero__text">
              Braillie teaches braille by touch. Your phone watches your finger on a printed sheet, and a patient voice tells you what you
              are feeling, a few letters at a time.
            </p>
            <div className="hero__actions">
              <Link className="cta" to="/login">
                Start learning {arrow}
              </Link>
              <a className="cta cta--outline" href="#how">
                How it works {arrow}
              </a>
            </div>
          </div>

          <div className={still ? 'world is-still' : 'world'} aria-hidden="true">
            <Reader at={i} />
            <p className="world__caption">
              <b>{letter.toUpperCase()}</b> dots {dots.join(', ')}
            </p>
          </div>
        </section>

        <section className="landing__section" id="how" aria-labelledby="how-h">
          <div className="sect-head">
            <h2 id="how-h">How it works</h2>
            <p>Print a sheet, point your phone at it, and feel the letters while Braillie says them.</p>
          </div>
          <HowItWorks />
        </section>

        <section className="landing__section landing__section--meet" id="braillo" aria-labelledby="meet-h">
          <MeetBraillo still={still} />
        </section>

        <section className="landing__section" id="sheets" aria-labelledby="sheets-h">
          <Sheets />
        </section>

        <section className="landing__section" id="made-for" aria-labelledby="made-h">
          <div className="sect-head">
            <h2 id="made-h">Made for blind and low-vision learners</h2>
            <p>Everything works by voice, by keyboard or with a screen reader.</p>
          </div>
          <MadeFor />
        </section>

        <section className="landing__end" aria-labelledby="end-h">
          <h2 id="end-h">Ready for your first letter?</h2>
          <p>It takes about two minutes to set up. Braillie talks you through every step.</p>
          <Link className="cta" to="/login">
            Start learning {arrow}
          </Link>
        </section>
      </main>

      <SiteFooter />
    </div>
  )
}

export default Landing
