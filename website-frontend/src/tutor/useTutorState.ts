import { useEffect, useState } from 'react'
import { TUTOR_API, type TutorState } from './tutorApi.ts'

export type Reach = 'checking' | 'yes' | 'no'

/** The tutor's current state, refreshed about every second. `reach` says whether the tutor could be reached at all. */
export function useTutorState(everyMs = 800): { state: TutorState | null; reach: Reach } {
  const [state, setState] = useState<TutorState | null>(null)
  const [reach, setReach] = useState<Reach>('checking')

  useEffect(() => {
    let stopped = false
    let timer = 0
    const poll = async () => {
      try {
        const res = await fetch(`${TUTOR_API}/api/state`, { cache: 'no-store' })
        const next: TutorState = await res.json()
        if (!stopped) {
          setState(next)
          setReach('yes')
        }
      } catch {
        if (!stopped) setReach('no')
      }
      if (!stopped) timer = window.setTimeout(poll, everyMs) // one request at a time: never piles up behind a slow answer
    }
    poll()
    return () => {
      stopped = true
      window.clearTimeout(timer)
    }
  }, [everyMs])

  return { state, reach }
}
