import { useCallback, useEffect, useRef, useState } from 'react'
import type { Reach } from '../tutor/useTutorState.ts'
import { setDialogue, type TutorState } from '../tutor/tutorApi.ts'
import { browserCanTalk, browserChannel, tutorCanTalk, tutorChannel, VoiceBlocked } from './channels.ts'
import { runLoginDialogue, type Actions, type Status } from './loginDialogue.ts'

const PREF = 'braillie.voice' // "off" once someone has turned the voice off: it is not started again for them

const wanted = (): boolean => {
  try {
    return window.localStorage.getItem(PREF) !== 'off'
  } catch {
    return true
  }
}
const remember = (on: boolean) => {
  try {
    window.localStorage.setItem(PREF, on ? 'on' : 'off')
  } catch {
    // just for this visit, then
  }
}

export type VoiceKind = 'tutor' | 'browser' | 'none'

export interface LoginVoice {
  kind: VoiceKind // who talks: the tutor on this computer, this browser, or nobody (nothing here can listen)
  running: boolean
  status: Status
  note: string // a problem to tell the person about (microphone refused, and so on)
  start(): void
  stop(): void // stop talking now (they are using the page instead)
  turnOff(): void // stop, and do not start by itself next time
}

interface Args {
  reach: Reach
  state: TutorState | null
  ready: boolean // who is here (if anyone) is known
  returning: string | null
  actions: Actions
}

/** The voice of the sign-in page. With the tutor running it starts by itself; with only the browser it starts when the person asks. */
export function useLoginVoice({ reach, state, ready, returning, actions }: Args): LoginVoice {
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState<Status>({ phase: 'idle' })
  const [note, setNote] = useState('')
  const abort = useRef<AbortController | null>(null)
  const latest = useRef({ returning, actions })
  const kind: VoiceKind = tutorCanTalk(state) ? 'tutor' : browserCanTalk() ? 'browser' : 'none'
  const kindNow = useRef(kind)
  useEffect(() => {
    latest.current = { returning, actions } // the conversation always uses the newest of these, without restarting
    kindNow.current = kind
  })

  const run = useCallback(async () => {
    if (abort.current) return
    const ac = new AbortController()
    abort.current = ac
    setRunning(true)
    setNote('')
    const viaTutor = kindNow.current === 'tutor'
    let ping = 0
    try {
      if (viaTutor) {
        await setDialogue(true).catch(() => {})
        ping = window.setInterval(() => void setDialogue(true).catch(() => {}), 10000)
      }
      await runLoginDialogue({
        channel: viaTutor ? tutorChannel() : browserChannel(),
        actions: {
          google: () => latest.current.actions.google(),
          guest: (n) => latest.current.actions.guest(n),
          carryOn: () => latest.current.actions.carryOn(),
          switchAccount: () => latest.current.actions.switchAccount(),
        },
        returning: latest.current.returning,
        signal: ac.signal,
        onStatus: setStatus,
      })
    } catch (e) {
      if (e instanceof VoiceBlocked) setNote('I cannot use the microphone. Allow it in the browser’s address bar, or use the buttons below.')
      else setNote('The voice stopped working. You can use the buttons below.')
    } finally {
      window.clearInterval(ping)
      if (viaTutor) void setDialogue(false).catch(() => {})
      if (abort.current === ac) abort.current = null
      setRunning(false)
      setStatus((s) => ({ ...s, phase: 'idle' }))
    }
  }, [])

  const stop = useCallback(() => {
    abort.current?.abort()
    window.speechSynthesis?.cancel()
  }, [])

  const turnOff = useCallback(() => {
    remember(false)
    stop()
  }, [stop])

  const start = useCallback(() => {
    remember(true)
    void run()
  }, [run])

  // With the tutor on this computer, talk straight away (a short wait first, so that a page that is loaded twice in development, or that
  // is left again at once, does not make the tutor speak twice).
  useEffect(() => {
    if (!ready || reach !== 'yes' || kind !== 'tutor' || !wanted()) return
    const t = window.setTimeout(() => void run(), 400)
    return () => window.clearTimeout(t)
  }, [ready, reach, kind, run])

  useEffect(
    () => () => {
      abort.current?.abort()
      window.speechSynthesis?.cancel()
    },
    [],
  )

  return { kind, running, status, note, start, stop, turnOff }
}
