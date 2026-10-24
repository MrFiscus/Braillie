// Following a spoken line as it is said, like the words of a song on screen: which sentence is being spoken now, and how far into it.
// The tutor says when the line started and how long its audio is; speech runs at a fairly even pace, so each word is given a share
// of that time by its length, with a little more after a comma and more again at the end of a sentence (where the voice pauses).

export interface Sentence {
  words: string[]
  start: number // where it begins, as a fraction (0..1) of the whole line
  end: number
  wordEnds: number[] // where each word ends, as a fraction of the whole line
}

const COMMA_PAUSE = 3 // characters' worth of time
const STOP_PAUSE = 7

/** "Welcome to Braillie. On this page..." -> one entry per sentence, each with its words and its share of the time. */
export function timeline(text: string): Sentence[] {
  const sentences = (text.match(/[^.?!]+[.?!]*(\s+|$)/g) ?? [text]).map((s) => s.trim()).filter(Boolean)
  const weighted = sentences.map((s) => {
    const words = s.split(/\s+/)
    const weights = words.map((w, i) => w.length + 1 + (/[,;:]$/.test(w) ? COMMA_PAUSE : 0) + (i === words.length - 1 ? STOP_PAUSE : 0))
    return { words, weights }
  })
  const total = weighted.reduce((n, s) => n + s.weights.reduce((a, b) => a + b, 0), 0) || 1
  let at = 0
  return weighted.map(({ words, weights }) => {
    const start = at / total
    const wordEnds = weights.map((w) => (at += w) / total)
    return { words, start, end: at / total, wordEnds }
  })
}

/** Where speech is, `fraction` (0..1) of the way through the line: the sentence, and how many of its words have been said. */
export function position(line: Sentence[], fraction: number): { sentence: number; said: number } {
  if (!line.length) return { sentence: 0, said: 0 }
  const f = Math.min(Math.max(fraction, 0), 1)
  const found = line.findIndex((s) => f < s.end)
  const i = found === -1 ? line.length - 1 : found // (at the very end: the last sentence)
  const s = line[i]
  return { sentence: i, said: f >= 1 ? s.words.length : s.wordEnds.filter((e) => e <= f).length }
}
