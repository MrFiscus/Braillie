// Who is using the app. A Google learner has an account and their progress is kept; a guest has just a first name for the greeting and
// nothing about them is saved anywhere. Pure helpers (tested in tests/appUser.test.mjs); the React hook is in useUser.ts.

export interface AppUser {
  kind: 'google' | 'guest'
  name: string // what the tutor calls them
  id?: string // google: the account id, the progress is kept under it. guest: an id for this visit only, so the tutor can tell one
  // guest from the next (each starts afresh, and is asked again how much braille they know), never saved anywhere
}

/** Where an older version kept the guest (this tab's sessionStorage). Guests are no longer kept anywhere: it is only cleared away. */
export const OLD_GUEST_KEY = 'braillie.guest'

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
  removeItem(key: string): void
}

const visitId = (): string => {
  try {
    return crypto.randomUUID().replace(/-/g, '')
  } catch {
    return `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`
  }
}

/** A new guest: a first name and an id for this visit. Kept in memory only, never stored: a reload or a new tab is the sign-in page again. */
export function newGuest(name: string): AppUser {
  return { kind: 'guest', name: firstName(name) || 'friend', id: visitId() }
}

/** Clear a guest an older version left in storage, so it cannot come back. */
export function forgetOldGuest(storage: StorageLike | null): void {
  try {
    storage?.removeItem(OLD_GUEST_KEY)
  } catch {
    // nothing to clear
  }
}

/** What to send the tutor for this user, and a key that changes only when the person does (so it is sent once, not on every render). */
export function sessionFor(user: AppUser): { body: { kind: 'google' | 'guest'; name: string; profile?: string }; key: string } {
  const body =
    user.kind === 'google' && user.id
      ? { kind: 'google' as const, name: user.name, profile: user.id }
      : { kind: 'guest' as const, name: user.name, ...(user.id ? { profile: user.id } : {}) } // (a guest's id only tells visits apart)
  return { body, key: `${body.kind}:${body.profile ?? ''}:${body.name}` }
}
