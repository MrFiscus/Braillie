import { useState } from 'react'
import { useSpeechPosition, type Speech } from '../voice/useSpeechPosition.ts'

/**
 * What Braillo is saying, one sentence at a time and large: the sentence being said, whole, fading in as the voice reaches it. Not word
 * by word: where each word falls can only be guessed (the voice gives no timings, and a line still arriving has only an estimated
 * length), and a guess that runs ahead and jumps back is worse to follow than a steady sentence. It never goes back to an earlier
 * sentence of the same line. Hidden from screen readers (they hear Braillie itself, and the same words read out over its voice would
 * make both hard to follow); the whole line is there for them as plain text instead.
 */
const Lyrics = ({ speech, speed = 1 }: { speech: Speech; speed?: number }) => {
  const { line, sentence } = useSpeechPosition(speech, speed)
  const [furthest, setFurthest] = useState({ text: speech.text, at: 0 })
  const at = furthest.text === speech.text ? Math.max(furthest.at, sentence) : sentence
  if (furthest.text !== speech.text || at > furthest.at) setFurthest({ text: speech.text, at }) // (React's "adjust state while rendering")

  const current = line[Math.min(at, line.length - 1)]
  return (
    <div className="lyrics">
      <p className="sr-only">{speech.text}</p>
      <div aria-hidden="true">
        <p className="lyrics__line" key={`${speech.text}#${at}`}>
          {current?.words.join(' ')}
        </p>
      </div>
    </div>
  )
}

export default Lyrics
