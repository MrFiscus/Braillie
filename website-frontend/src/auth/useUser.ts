import { useCallback, useEffect, useState } from 'react'
import { supabase } from '../auths/supabaseClients.ts'
import { clearGuest, loadGuest, saveGuest, userFromAccount, type AppUser } from './appUser.ts'

const store = (): Storage | null => {
  try {
    return window.sessionStorage
  } catch {
    return null
  }
}

/**
 * The person using the app: a Google account (from Supabase) or a guest (a first name kept for this tab). `loading` is true until we
 * know, so a page does not send someone to the sign-in page while their account is still being looked up.
 */
export function useUser(): { user: AppUser | null; loading: boolean; continueAsGuest: (name: string) => AppUser; signOut: () => Promise<void> } {
  const [user, setUser] = useState<AppUser | null>(() => loadGuest(store()))
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    supabase.auth
      .getSession()
      .then(({ data }) => {
        if (alive && data.session?.user) setUser(userFromAccount(data.session.user))
      })
      .catch(() => {}) // no account service reachable: a guest can still continue
      .finally(() => alive && setLoading(false))
    const { data } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!alive) return
      if (session?.user) setUser(userFromAccount(session.user))
      else setUser(loadGuest(store()))
    })
    return () => {
      alive = false
      data.subscription.unsubscribe()
    }
  }, [])

  const continueAsGuest = useCallback((name: string) => {
    const guest = saveGuest(store(), name)
    setUser(guest)
    return guest
  }, [])

  const signOut = useCallback(async () => {
    clearGuest(store())
    setUser(null)
    try {
      await supabase.auth.signOut()
    } catch {
      // nothing else to do
    }
  }, [])

  return { user, loading, continueAsGuest, signOut }
}
