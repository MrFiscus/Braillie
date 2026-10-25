import { useEffect, useState } from 'react'
import type { TutorState } from '../tutor/tutorApi.ts'
import type { Speech } from './useSpeechPosition.ts'

const HOLD_MS = 900 // a caption stays this long after its line ends, so a short gap (between two lines, or one missed check) does not flash

/**
 * The line Braillo is saying, for the captions: the one playing now, or the one just finished for a moment more. Without the hold the
 * caption blinks to the hint and back whenever one check of the tutor falls between two lines.
 */
export function useSpokenLine(state: TutorState | null): { speaking: boolean; speech: Speech | null } {
  const live = state?.speaking && state?.speech ? state.speech : null
  const [held, setHeld] = useState<Speech | null>(null)

  // when a line stops being the live one, it is held (and the hold runs out by itself)
  useEffect(() => {
    if (!live) return
    return () => setHeld(live)
  }, [live])
  useEffect(() => {
    if (!held) return
    const t = window.setTimeout(() => setHeld(null), HOLD_MS)
    return () => window.clearTimeout(t)
  }, [held])

  const speech = live ?? held
  return { speaking: !!speech, speech }
}
