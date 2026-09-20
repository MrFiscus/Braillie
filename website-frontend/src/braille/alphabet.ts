// Standard English braille: which of the six dot positions are raised for each letter. Dots are numbered down the left (1, 2, 3) then
// down the right (4, 5, 6). Used only to DRAW braille (the wordmark, the mode icons); the tutor does the actual reading.
export const DOTS: Record<string, number[]> = {
  a: [1], b: [1, 2], c: [1, 4], d: [1, 4, 5], e: [1, 5], f: [1, 2, 4], g: [1, 2, 4, 5], h: [1, 2, 5], i: [2, 4], j: [2, 4, 5],
  k: [1, 3], l: [1, 2, 3], m: [1, 3, 4], n: [1, 3, 4, 5], o: [1, 3, 5], p: [1, 2, 3, 4], q: [1, 2, 3, 4, 5], r: [1, 2, 3, 5],
  s: [2, 3, 4], t: [2, 3, 4, 5], u: [1, 3, 6], v: [1, 2, 3, 6], w: [2, 4, 5, 6], x: [1, 3, 4, 6], y: [1, 3, 4, 5, 6], z: [1, 3, 5, 6],
}

/** Where a dot sits inside a cell that is `w` wide and `h` tall (2 columns by 3 rows, with a margin). */
export function dotCentre(n: number, w: number, h: number): { x: number; y: number } {
  const col = n <= 3 ? 0 : 1
  const row = (n - 1) % 3
  return { x: w * (col === 0 ? 0.3 : 0.7), y: h * (0.2 + 0.3 * row) }
}
