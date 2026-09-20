// Words for a braille cell's raised dots. Dots are numbered down the left (1, 2, 3) then down the right (4, 5, 6).
const POSITIONS = ['top-left', 'middle-left', 'bottom-left', 'top-right', 'middle-right', 'bottom-right']

export function describeDots(dots: number[]): string {
  if (dots.length === 0) return 'no raised dots'
  const where = [...dots].sort((a, b) => a - b).map((d) => POSITIONS[d - 1])
  const list = where.length === 1 ? where[0] : `${where.slice(0, -1).join(', ')} and ${where[where.length - 1]}`
  return `${dots.length === 1 ? 'one dot' : `${dots.length} dots`}: ${list}`
}
