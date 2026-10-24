import { useEffect, useMemo, useState } from 'react'
import { position, timeline, type Sentence } from './lyrics.ts'

export interface Speech {
  text: string
  started: number // when it began playing (seconds, the tutor's clock: this computer's)
  seconds: number // how long its audio is at normal speed
}

/** Where Braillo is in the line it is saying, updated every frame: the sentences, which one is being said, and how many of its words.
 * `null` speech: nothing is being said. */
export function useSpeechPosition(speech: Speech | null | undefined, speed = 1): { line: Sentence[]; sentence: number; said: number } {
  const text = speech?.text ?? ''
  const line = useMemo(() => (text ? timeline(text) : []), [text])
  const [now, setNow] = useState(() => Date.now() / 1000)

  useEffect(() => {
    if (!speech) return
    let frame = 0
    const tick = () => {
      setNow(Date.now() / 1000)
      frame = window.requestAnimationFrame(tick)
    }
    frame = window.requestAnimationFrame(tick)
    return () => window.cancelAnimationFrame(frame)
  }, [speech])

  if (!speech) return { line, sentence: 0, said: 0 }
  const length = speech.seconds > 0 ? speech.seconds / Math.max(speed, 0.1) : 0
  return { line, ...position(line, length ? (now - speech.started) / length : 0) }
}
