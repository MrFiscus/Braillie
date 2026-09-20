// One braille cell drawn as two columns of three dots. A raised dot is a solid disc, a flat position a hollow ring, so the pattern reads
// without colour. Dots are numbered down the left (1, 2, 3) then the right (4, 5, 6).
import { describeDots } from '../tutor/describeDots.ts'
import { dotCentre } from '../braille/alphabet.ts'

interface DotCellProps {
  dots: number[]
  size?: number // height in px (it scales with the page through the svg viewBox)
  tone?: 'ink' | 'accent'
  label?: string // what a screen reader says (defaults to the dots)
  decorative?: boolean // an icon next to text that already says it: hidden from screen readers
}

const W = 62
const H = 100

const DotCell = ({ dots, size = 120, tone = 'ink', label, decorative = false }: DotCellProps) => (
  <svg
    className="braillecell"
    width={(size * W) / H}
    height={size}
    viewBox={`0 0 ${W} ${H}`}
    {...(decorative ? { 'aria-hidden': true } : { role: 'img', 'aria-label': label ?? `Braille cell, ${describeDots(dots)}` })}
  >
    <rect className="cell-frame" x={1.5} y={1.5} width={W - 3} height={H - 3} rx={13} />
    {[1, 2, 3, 4, 5, 6].map((n) => {
      const { x, y } = dotCentre(n, W, H)
      const on = dots.includes(n)
      return <circle key={n} cx={x} cy={y} r={on ? 9.5 : 6} className={on ? (tone === 'accent' ? 'cell-dot--accent' : 'cell-dot--on') : 'cell-dot--off'} />
    })}
  </svg>
)

export default DotCell
