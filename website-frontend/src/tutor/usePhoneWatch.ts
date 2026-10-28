import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import type { TutorState } from './tutorApi.ts'
import { RECONNECT_AFTER_MS, shouldReconnect } from './phoneWatch.ts'

// The phone is the camera (and often the speaker and microphone). If it drops out for good while it is needed, Braillo says to scan the
// code again, so the code has to be on the screen: after a while without it, the Connect Phone page opens (and, once the phone is back,
// returns to where they were, the lesson included). A short wobble (a frozen camera that restarts, a Wi-Fi hiccup) does not count.

export function usePhoneWatch(state: TutorState | null, pathname: string): void {
  const navigate = useNavigate()
  const linked = useRef(false)
  const connected = state?.phone ? state.phone.connected : undefined
  useEffect(() => {
    if (connected) linked.current = true
  }, [connected])
  useEffect(() => {
    if (connected !== false || !shouldReconnect(pathname, linked.current, false, RECONNECT_AFTER_MS)) return
    const t = window.setTimeout(() => navigate(`/connect-phone?back=${encodeURIComponent(pathname)}`), RECONNECT_AFTER_MS)
    return () => window.clearTimeout(t) // (back before then: nothing to do)
  }, [connected, pathname, navigate])
}
