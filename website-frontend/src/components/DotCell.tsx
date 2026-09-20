// One braille cell drawn as two columns of three dots, raised dots filled. Dots are numbered down the left (1, 2, 3) then the right (4, 5, 6).
import { describeDots } from '../tutor/describeDots.ts'

interface DotCellProps {
  dots: number[]
  size?: number // height in px
  raised?: string // colour of a raised dot
  flat?: string // colour of an empty position
  label?: string // what a screen reader says (defaults to the dots)
}

const DotCell = ({ dots, size = 120, raised = '#a855f7', flat = 'rgba(148, 163, 184, 0.25)', label }: DotCellProps) => {
  const w = size * 0.62
  const r = size * 0.11
  const x = [w * 0.28, w * 0.72]
  const y = [size * 0.2, size * 0.5, size * 0.8]
  return (
    <svg width={w} height={size} viewBox={`0 0 ${w} ${size}`} role="img" aria-label={label ?? `Braille cell, ${describeDots(dots)}`} style={{ flexShrink: 0 }}>
      <rect x={1} y={1} width={w - 2} height={size - 2} rx={size * 0.12} fill="none" stroke="rgba(148, 163, 184, 0.35)" strokeWidth={2} />
      {[1, 2, 3, 4, 5, 6].map((n) => {
        const col = n <= 3 ? 0 : 1
        const row = (n - 1) % 3
        const on = dots.includes(n)
        return <circle key={n} cx={x[col]} cy={y[row]} r={r} fill={on ? raised : flat} stroke={on ? raised : 'none'} />
      })}
    </svg>
  )
}

export default DotCell
