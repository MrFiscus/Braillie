// Two ways for the sign-in page to talk with someone.
//   tutorChannel:   the tutor on this computer speaks (Deepgram) and listens (its microphone, Deepgram). It is the same voice as the rest of
//                   the app, needs no permission from the browser, and works without anyone pressing anything first.
//   browserChannel: the browser's own speech and speech recognition, for when the tutor is not running. Chrome, Edge and Safari have them;
//                   the browser asks for the microphone once, and a person has to press a button first (browsers do not speak unprompted).
import { fetchState, sayPrompt, type TutorState } from '../tutor/tutorApi.ts'
import { PROMPT_TEXT, nextVersion, type Channel, type PromptName } from './loginDialogue.ts'

const sleep = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms))

/** Can the tutor speak AND listen? (It listens only with a live Deepgram microphone: `heard` is null otherwise.) */
export const tutorCanTalk = (state: TutorState | null): boolean => !!state?.hub && state.heard != null && state.config.voice?.ok !== false

const POLL_MS = 200
const LISTEN_MS = 12000

// The tutor's microphone stays open while it talks (it filters out its own voice), so an answer can come at any moment: while the
// question is still being asked (that also cuts the question short), or after. Answers are counted from the moment a line starts.
export function tutorChannel(): Channel {
  let since: number | null = null // what the microphone had heard when the last line started; anything after is an answer
  let overSpeech = false // the last answer was a word or two caught while the tutor was still talking
  const heardNow = async () => (await fetchState().catch(() => null))?.heard?.n ?? 0

  return {
    async say(prompt, who) {
      since = await heardNow()
      await sayPrompt(prompt, who, true) // a new line replaces anything still being said (they answered over it)
      const began = Date.now()
      let started = false
      for (;;) {
        let state
        try {
          state = await fetchState()
        } catch {
          break // cannot ask: do not hold the conversation up
        }
        if ((state.heard?.n ?? 0) > since) break // they answered over it: go and hear the answer
        if (state.speaking) started = true
        else if (started || Date.now() - began > 2500) break // finished; or it never started talking (say nothing more about it)
        await sleep(POLL_MS)
      }
    },

    async listen(signal) {
      const ends = Date.now() + LISTEN_MS
      const seen = since ?? (await heardNow())
      since = null
      while (!signal.aborted && Date.now() < ends) {
        const heard = (await fetchState().catch(() => null))?.heard
        if (heard && heard.n > seen && heard.text) {
          overSpeech = heard.reason === 'heard over speech'
          return heard.text
        }
        await sleep(POLL_MS)
      }
      return null
    },

    heardOverSpeech: () => overSpeech,
  }
}

// ---- the browser's own voice ------------------------------------------------------------------------------------------------------
interface Recogniser {
  lang: string
  interimResults: boolean
  maxAlternatives: number
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null
  onerror: ((e: { error: string }) => void) | null
  onend: (() => void) | null
  start(): void
  stop(): void
  abort(): void
}
type RecogniserCtor = new () => Recogniser

const recogniserCtor = (): RecogniserCtor | null => {
  const w = window as unknown as { SpeechRecognition?: RecogniserCtor; webkitSpeechRecognition?: RecogniserCtor }
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null
}

/** Can this browser both speak and listen? */
export const browserCanTalk = (): boolean => typeof window !== 'undefined' && 'speechSynthesis' in window && recogniserCtor() !== null

/** The microphone was refused (or there is none): the conversation cannot go on by voice. */
export class VoiceBlocked extends Error {}

export function browserChannel(): Channel {
  return {
    async say(prompt: PromptName, who) {
      const text = PROMPT_TEXT[prompt](who ?? '')
      nextVersion(prompt) // (next time this line is said, it comes in other words)
      window.speechSynthesis.cancel()
      // sentence by sentence: some browsers cut off one long utterance
      for (const sentence of text.match(/[^.?!]+[.?!]?/g) ?? [text]) {
        await new Promise<void>((done) => {
          const u = new SpeechSynthesisUtterance(sentence.trim())
          u.rate = 0.95
          u.onend = () => done()
          u.onerror = () => done()
          window.speechSynthesis.speak(u)
        })
      }
      await sleep(250)
    },

    listen(signal) {
      const Ctor = recogniserCtor()
      if (!Ctor) return Promise.reject(new VoiceBlocked('This browser cannot listen.'))
      return new Promise<string | null>((resolve, reject) => {
        const r = new Ctor()
        let heard: string | null = null
        r.lang = navigator.language || 'en-US'
        r.interimResults = false
        r.maxAlternatives = 1
        r.onresult = (e) => {
          heard = e.results[0]?.[0]?.transcript ?? null
        }
        r.onerror = (e) => {
          if (e.error === 'not-allowed' || e.error === 'service-not-allowed' || e.error === 'audio-capture') reject(new VoiceBlocked(e.error))
        }
        const stopper = window.setTimeout(() => r.stop(), LISTEN_MS)
        const abort = () => r.abort()
        signal.addEventListener('abort', abort)
        r.onend = () => {
          window.clearTimeout(stopper)
          signal.removeEventListener('abort', abort)
          resolve(heard)
        }
        try {
          r.start()
        } catch {
          resolve(null)
        }
      })
    },
  }
}
