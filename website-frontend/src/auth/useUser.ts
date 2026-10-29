import { useCallback, useEffect, useState, useSyncExternalStore } from 'react'
import { supabase } from '../auths/supabaseClients.ts'
import { forgetOldGuest, newGuest, userFromAccount, type AppUser } from './appUser.ts'

// The guest, for this visit only: shared by every page (so moving between pages keeps them) but never stored, so a reload, a restart
// or a new tab never brings an old guest back.
let guest: AppUser | null = null
const listeners = new Set<() => void>()
const setGuest = (g: AppUser | null) => {
  guest = g
  listeners.forEach((l) => l())
}
const watchGuest = (l: () => void) => {
  listeners.add(l)
  return () => void listeners.delete(l)
}

try {
  forgetOldGuest(window.sessionStorage)
} catch {
  // no storage: nothing old to clear
}

/**
 * The person using the app: a Google account (from Supabase) or a guest (a first name, for this visit only). `loading` is true until we
 * know, so a page does not send someone to the sign-in page while their account is still being looked up.
 */
export function useUser(): { user: AppUser | null; loading: boolean; continueAsGuest: (name: string) => AppUser; signOut: () => Promise<void> } {
  const visitor = useSyncExternalStore(watchGuest, () => guest)
  const [account, setAccount] = useState<AppUser | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    supabase.auth
      .getSession()
      .then(({ data }) => {
        if (alive) setAccount(data.session?.user ? userFromAccount(data.session.user) : null)
      })
      .catch(() => {}) // no account service reachable: a guest can still continue
      .finally(() => alive && setLoading(false))
    const { data } = supabase.auth.onAuthStateChange((_event, session) => {
      if (alive) setAccount(session?.user ? userFromAccount(session.user) : null)
    })
    return () => {
      alive = false
      data.subscription.unsubscribe()
    }
  }, [])

  const continueAsGuest = useCallback((name: string) => {
    const g = newGuest(name)
    setGuest(g)
    return g
  }, [])

  const signOut = useCallback(async () => {
    setGuest(null)
    setAccount(null)
    try {
      await supabase.auth.signOut()
    } catch {
      // nothing else to do
    }
  }, [])

  return { user: account ?? visitor, loading, continueAsGuest, signOut }
}
