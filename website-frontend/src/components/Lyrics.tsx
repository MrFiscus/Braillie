import { useSpeechPosition, type Speech } from '../voice/useSpeechPosition.ts'

/**
 * What Braillo is saying, one sentence at a time and large, with each word filling in as it is spoken, and the next sentence waiting
 * underneath: so you can follow along, and see how fast it is talking. Hidden from screen readers (they hear Braillie itself, and the
 * same words read out over its voice would make both hard to follow); the whole line is there for them as plain text instead.
 */
const Lyrics = ({ speech, speed = 1 }: { speech: Speech; speed?: number }) => {
  const { line, sentence, said } = useSpeechPosition(speech, speed)
  const current = line[sentence]
  const next = line[sentence + 1]

  return (
    <div className="lyrics">
      <p className="sr-only">{speech.text}</p>
      <div aria-hidden="true">
        {line.length > 1 && (
          <p className="lyrics__count">
            {sentence + 1} of {line.length}
          </p>
        )}
        <p className="lyrics__line" key={sentence}>
          {current?.words.map((w, i) => (
            <span key={i} className={i < said ? 'is-said' : i === said ? 'is-now' : undefined}>
              {w}{' '}
            </span>
          ))}
        </p>
        {next && <p className="lyrics__next">{next.words.join(' ')}</p>}
      </div>
    </div>
  )
}

export default Lyrics
