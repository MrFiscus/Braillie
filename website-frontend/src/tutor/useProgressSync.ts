import { useEffect, useRef, useState } from 'react'
import { supabase } from '../auths/supabaseClients.ts'
import { getProgress, mergeProgress, type TutorState } from './tutorApi.ts'
import { ProgressSync, progressSignature, supabaseAccount, type SupabaseLike, type SyncStatus } from './progressSync.ts'

/**
 * Keeps the learner's progress in their account as well as on the tutor's computer. Does nothing unless the tutor is in learn mode
 * (it reports progress) and does nothing harmful if nobody is signed in or the account cannot be reached.
 */
export function useProgressSync(state: TutorState | null): { status: SyncStatus; detail: string } {
  const [info, setInfo] = useState<{ status: SyncStatus; detail: string }>({ status: 'idle', detail: '' })
  const sync = useRef<ProgressSync | null>(null)
  const lastSignature = useRef('')
  const learning = !!state?.progress
  const signature = progressSignature(state?.progress)

  useEffect(() => {
    if (!learning) return
    const s = new ProgressSync(supabaseAccount(supabase as unknown as SupabaseLike), { get: getProgress, merge: mergeProgress }, (status, detail) => setInfo({ status, detail }))
    sync.current = s
    void s.start().then(() => {
      lastSignature.current = '' // whatever the account brought in has been merged: what the tutor holds now is worth saving back
    })
    return () => {
      s.dispose()
      sync.current = null
    }
  }, [learning])

  useEffect(() => {
    if (!sync.current || !signature || signature === lastSignature.current) return
    lastSignature.current = signature
    sync.current.schedule()
  }, [signature])

  return info
}
