import { useEffect, useRef } from 'react'
import type { AppUser } from '../auth/appUser.ts'
import { sessionFor } from '../auth/appUser.ts'
import { forgetSession, setSession, type TutorState } from './tutorApi.ts'

/**
 * Keeps the tutor in step with who is using the website (it runs on every page): it is told about the person when it does not know them
 * yet (or was restarted and forgot), and when the person changes. When nobody is signed in here but the tutor still has someone (an old
 * guest from before a reload, say), it is told to forget them. `user` undefined: this tab says nothing. Does nothing until the tutor answers and we know who the user is.
 */
export function useTutorSession(user: AppUser | null | undefined, loading: boolean, state: TutorState | null): void {
  const sent = useRef('') // the person we last told it about
  const inflight = useRef(false)
  const hub = state?.hub
  const tutorKnowsSomeone = !!hub?.user
  useEffect(() => {
    if (user === undefined || !hub || loading || inflight.current) return
    if (!user) {
      if (!tutorKnowsSomeone) return
      inflight.current = true
      forgetSession()
        .then(() => {
          sent.current = ''
        })
        .catch(() => {}) // the next update tries again
        .finally(() => {
          inflight.current = false
        })
      return
    }
    const { body, key } = sessionFor(user)
    if (tutorKnowsSomeone && sent.current === key) return
    inflight.current = true
    setSession(body)
      .then(() => {
        sent.current = key
      })
      .catch(() => {}) // the next update tries again
      .finally(() => {
        inflight.current = false
      })
  }, [user, loading, hub, tutorKnowsSomeone])
}
