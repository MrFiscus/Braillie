// A word written in braille dots, as a picture: the site's one piece of decoration. It is hidden from screen readers (the word is already
// on the page in plain text); `caption` says what it is for anyone who can see it.
import { DOTS, dotCentre } from '../braille/alphabet.ts'

const CELL_W = 34
const CELL_H = 54
const GAP = 14

const BrailleWord = ({ word, caption }: { word: string; caption?: string }) => {
  const letters = [...word.toLowerCase()].filter((l) => DOTS[l])
  const width = letters.length * CELL_W + (letters.length - 1) * GAP
  return (
    <figure className="wordmark" aria-hidden="true">
      <svg viewBox={`0 0 ${width} ${CELL_H}`} focusable="false">
        {letters.map((l, i) => {
          const ox = i * (CELL_W + GAP)
          return (
            <g key={i} transform={`translate(${ox} 0)`}>
              {[1, 2, 3, 4, 5, 6].map((n) => {
                const { x, y } = dotCentre(n, CELL_W, CELL_H)
                const on = DOTS[l].includes(n)
                return <circle key={n} cx={x} cy={y} r={on ? 5.6 : 2.4} className={on ? (i === 0 ? 'cell-dot--accent' : 'cell-dot--on') : 'cell-dot--off'} />
              })}
            </g>
          )
        })}
      </svg>
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  )
}

export default BrailleWord
