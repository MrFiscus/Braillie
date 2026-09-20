// Keeps the learner's progress in their account (Supabase) as well as on the tutor's computer, without either overwriting the other.
//
//   start(): fetch the saved copy from the account and MERGE it into the tutor's (the merge keeps the more recently practised letter, the
//            larger counts and the better lesson scores, and merging the same copy twice changes nothing)
//   schedule(): after progress changes, save the tutor's copy to the account, at most once every few seconds
//
// Pure logic: the account and the tutor are passed in, so it is tested with fakes (tests/progressSync.test.mjs) and works with any store.
import type { ProgressSummary } from './tutorApi.ts'

export type SyncStatus = 'idle' | 'signed-out' | 'syncing' | 'saved' | 'error'

export interface Account {
  /** The signed-in learner's id, or null when nobody is signed in. */
  userId(): Promise<string | null>
  /** The saved progress for that learner, or null if there is none yet. */
  load(userId: string): Promise<unknown | null>
  save(userId: string, data: unknown): Promise<void>
}

export interface TutorProgress {
  get(): Promise<unknown>
  merge(data: unknown): Promise<void>
}

/** A short string that changes whenever the progress does (so the page knows when to save). */
export function progressSignature(p: ProgressSummary | null | undefined): string {
  if (!p) return ''
  const mastery = Object.entries(p.mastery).map(([l, m]) => `${l}${m}`).join(',')
  const lessons = Object.entries(p.lessons).map(([id, v]) => `${id}:${v.best}:${v.times}`).join(',')
  return [p.sessions, p.practised, p.learned, p.streak, mastery, lessons, p.confusions.map((c) => `${c.touched}${c.wanted}${c.count}`).join(',')].join('|')
}

export class ProgressSync {
  status: SyncStatus = 'idle'
  detail = ''
  private timer: ReturnType<typeof setTimeout> | null = null
  private saving = false
  private again = false
  private started = false

  private account: Account
  private tutor: TutorProgress
  private onChange: (status: SyncStatus, detail: string) => void
  private delayMs: number

  constructor(account: Account, tutor: TutorProgress, onChange: (status: SyncStatus, detail: string) => void = () => {}, delayMs = 3000) {
    this.account = account
    this.tutor = tutor
    this.onChange = onChange
    this.delayMs = delayMs
  }

  private set(status: SyncStatus, detail = '') {
    this.status = status
    this.detail = detail
    this.onChange(status, detail)
  }

  /** Once per session: bring the account's saved copy into the tutor. Safe to call again (does nothing the second time). */
  async start(): Promise<SyncStatus> {
    if (this.started) return this.status
    this.started = true
    try {
      const id = await this.account.userId()
      if (!id) {
        this.set('signed-out', 'Progress is kept on this computer only. Sign in to keep it in your account.')
        return this.status
      }
      this.set('syncing')
      const saved = await this.account.load(id)
      if (saved) await this.tutor.merge(saved)
      this.set('saved', 'Progress is kept in your account.')
    } catch (e) {
      this.set('error', `Could not reach your account (${e instanceof Error ? e.message : String(e)}). Progress is still kept on this computer.`)
    }
    return this.status
  }

  /** Progress changed: save it soon (waits a few seconds so a burst of answers is one save). */
  schedule(): void {
    if (this.status === 'signed-out') return
    if (this.timer) clearTimeout(this.timer)
    this.timer = setTimeout(() => void this.push(), this.delayMs)
  }

  /** Save the tutor's copy to the account now. One save at a time; a change during a save triggers one more afterwards. */
  async push(): Promise<SyncStatus> {
    if (this.timer) {
      clearTimeout(this.timer)
      this.timer = null
    }
    if (this.saving) {
      this.again = true
      return this.status
    }
    this.saving = true
    try {
      const id = await this.account.userId()
      if (!id) {
        this.set('signed-out', 'Progress is kept on this computer only. Sign in to keep it in your account.')
        return this.status
      }
      this.set('syncing')
      await this.account.save(id, await this.tutor.get())
      this.set('saved', 'Progress is kept in your account.')
    } catch (e) {
      this.set('error', `Could not save to your account (${e instanceof Error ? e.message : String(e)}). Progress is still kept on this computer.`)
    } finally {
      this.saving = false
      if (this.again) {
        this.again = false
        this.schedule()
      }
    }
    return this.status
  }

  dispose(): void {
    if (this.timer) clearTimeout(this.timer)
    this.timer = null
  }
}

/** The account as a Supabase table `learning_progress` (see braille_tutor/supabase_progress.sql). Only the parts of the client used. */
export interface SupabaseLike {
  auth: { getSession(): Promise<{ data: { session: { user: { id: string } } | null } }> }
  from(table: string): {
    select(columns: string): { eq(column: string, value: string): { maybeSingle(): Promise<{ data: { data: unknown } | null; error: { message: string } | null }> } }
    upsert(row: Record<string, unknown>): Promise<{ error: { message: string } | null }>
  }
}

export function supabaseAccount(supabase: SupabaseLike, table = 'learning_progress'): Account {
  return {
    async userId() {
      const { data } = await supabase.auth.getSession()
      return data.session?.user.id ?? null
    },
    async load(userId) {
      const { data, error } = await supabase.from(table).select('data').eq('user_id', userId).maybeSingle()
      if (error) throw new Error(error.message)
      return data?.data ?? null
    },
    async save(userId, data) {
      const { error } = await supabase.from(table).upsert({ user_id: userId, data, updated_at: new Date().toISOString() })
      if (error) throw new Error(error.message)
    },
  }
}
