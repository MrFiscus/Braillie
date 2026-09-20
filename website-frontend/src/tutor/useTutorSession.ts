import { useEffect, useRef } from 'react'
import type { AppUser } from '../auth/appUser.ts'
import { sessionFor } from '../auth/appUser.ts'
import { setSession, type TutorState } from './tutorApi.ts'

/**
 * Tells the tutor who is using it: when it does not know anyone yet (or was restarted and forgot), and when the person changes. A
 * Google learner's progress is then kept; a guest's never is. Does nothing until the tutor answers and we know who the user is.
 */
export function useTutorSession(user: AppUser | null, state: TutorState | null): void {
  const sent = useRef('') // the person we last told it about
  const inflight = useRef(false)
  const hub = state?.hub
  const tutorKnowsSomeone = !!hub?.user
  useEffect(() => {
    if (!user || !hub || inflight.current) return
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
  }, [user, hub, tutorKnowsSomeone])
}
