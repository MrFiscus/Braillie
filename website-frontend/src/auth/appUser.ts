// Who is using the app. A Google learner has an account and their progress is kept; a guest has just a first name for the greeting and
// nothing about them is saved anywhere. Pure helpers (tested in tests/appUser.test.mjs); the React hook is in useUser.ts.

export interface AppUser {
  kind: 'google' | 'guest'
  name: string // what the tutor calls them
  id?: string // the account id (google only): the progress is kept under it
}

export const GUEST_KEY = 'braillie.guest'

/** The name to greet someone by: their first name, tidied. */
export function firstName(full: string | null | undefined): string {
  const tidy = (full ?? '').replace(/[^\p{L}\p{N} .'-]/gu, '').replace(/\s+/g, ' ').trim()
  return tidy.split(' ')[0]?.slice(0, 40) ?? ''
}

interface SupabaseUserLike {
  id: string
  email?: string | null
  user_metadata?: Record<string, unknown> | null
}

/** A signed-in Google account as an AppUser: given name, else the first word of the full name, else the start of the email. */
export function userFromAccount(u: SupabaseUserLike): AppUser {
  const meta = u.user_metadata ?? {}
  const pick = (k: string) => (typeof meta[k] === 'string' ? (meta[k] as string) : '')
  const name = firstName(pick('given_name')) || firstName(pick('full_name')) || firstName(pick('name')) || firstName((u.email ?? '').split('@')[0]) || 'friend'
  return { kind: 'google', name, id: u.id }
}

interface StorageLike {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

/** A guest, remembered for this browser tab only (sessionStorage), so a page refresh does not ask again. Never anything but the name. */
export function loadGuest(storage: StorageLike | null): AppUser | null {
  try {
    const raw = storage?.getItem(GUEST_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    const name = firstName(typeof parsed?.name === 'string' ? parsed.name : '')
    return name ? { kind: 'guest', name } : null
  } catch {
    return null
  }
}

export function saveGuest(storage: StorageLike | null, name: string): AppUser {
  const user: AppUser = { kind: 'guest', name: firstName(name) || 'friend' }
  try {
    storage?.setItem(GUEST_KEY, JSON.stringify({ name: user.name }))
  } catch {
    // storage may be blocked: the guest still continues, they are just asked again after a refresh
  }
  return user
}

export function clearGuest(storage: StorageLike | null): void {
  try {
    storage?.removeItem(GUEST_KEY)
  } catch {
    // nothing to clear
  }
}

/** What to send the tutor for this user, and a key that changes only when the person does (so it is sent once, not on every render). */
export function sessionFor(user: AppUser): { body: { kind: 'google' | 'guest'; name: string; profile?: string }; key: string } {
  const body = user.kind === 'google' && user.id ? { kind: 'google' as const, name: user.name, profile: user.id } : { kind: 'guest' as const, name: user.name }
  return { body, key: `${body.kind}:${body.profile ?? ''}:${body.name}` }
}
