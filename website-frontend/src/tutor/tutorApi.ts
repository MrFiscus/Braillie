// Talks to the braille tutor (braille_tutor/tutor_server.py, started on the laptop). The API is documented in braille_tutor/API.md.
// Override the address with VITE_TUTOR_API (e.g. in .env.local) if the tutor is not on this computer's port 8000.
export const TUTOR_API: string = import.meta.env.VITE_TUTOR_API ?? 'http://127.0.0.1:8000'

export interface PhoneInfo {
  connected: boolean
  address: string
  code: string
  qr: string // path of the QR code image, relative to TUTOR_API
  tunnel?: boolean // reached through a public tunnel (--phone-tunnel, or the backup link once it is in use)
  backup?: boolean // a backup link (a tunnel) is ready, for a Wi-Fi that blocks the phone: see switchToBackupLink
  trusted?: boolean // the phone gets a real certificate (--phone-trusted, --phone-domain or --phone-tunnel): no "connection is not private" page to click through
  instructions: string
  diagnosis?: string | null // why no video has arrived yet, in plain words
  sound?: boolean
  mic?: boolean
}

export type LearningPhase = 'placing' | 'idle' | 'await_sheet' | 'warmup' | 'teach' | 'practice' | 'review' | 'word' | 'recap' | 'explore' | 'done' // placing: a new learner is asked how much braille they know; warmup: letters that went wrong before, taught again; word: spelling a real word at the end of a lesson

/** Where the learner is in the guided lessons (only when the tutor runs with --mode learn). */
export interface LearningStatus {
  phase: LearningPhase
  lesson: { id: string; title: string; index: number; of: number; sheet: string } | null
  target: string | null // the letter being asked for
  word?: string | null // while spelling a word at the end of a lesson: the word
  correct?: number // right answers this session (counts up: Braillo hops for joy at each)
  target_dots: number[]
  target_label?: string | null // how the sheet labels it: the letter, a digit, or NUM, CAP and the punctuation marks
  remaining: number // letters still to do in this part of the lesson, including the current one
  letters: number // letters in the lesson
  in_a_row: number
  hints: number
  tries: number
  round: number
}

/** What has been learned so far. `mastery` is 0 (never practised) to 1 (solid) per letter. */
export interface ProgressSummary {
  learned: number
  practised: number
  sessions: number
  streak: number
  best_streak: number
  mastery: Record<string, number>
  confusions: { touched: string; wanted: string; count: number }[]
  lessons: Record<string, { best: number; last: number; times: number }>
  level?: Level | '' // how much braille they said they knew before their first lesson ('' if not asked)
}

export type HubMode = 'menu' | 'learn' | 'read' | 'quiz'

/** The menu-driven tutor: who is using it and which activity they chose (null when the tutor was started in a fixed mode). */
export interface HubInfo {
  mode: HubMode
  modes: Record<string, { title: string; sheet: string }>
  user: { name: string; kind: 'google' | 'guest'; saves: boolean } | null
  greeted: boolean
  next_lesson?: { id: string; title: string; index: number; of: number; letters: string[] } | null // what "Continue" starts; null when all are done
}

export interface TutorState {
  config: {
    mode: string
    commands: string[]
    sheet?: string
    voice?: { ok: boolean; problems: string[]; warnings: string[] } | null
  }
  camera: { ok: boolean; frames: number }
  phone: PhoneInfo | null // null unless the tutor was started with --phone-camera
  page: { ok: boolean; message: string }
  tutor: { mode: string; state: string; prompt: string; question?: number; total?: number; asked?: number; correct?: number; tries?: number }
  finger: {
    page_mm: [number, number] | null
    source?: 'camera' | 'posted' | null // "posted": a click on the video is standing in for the camera's tracking, briefly
    cell: { letter: string | null; label: string | null; name: string | null; dots: number[] } | null
  }
  said: { t: number; kind: string; text: string }[]
  reading?: { locked: number; total: number; between_pages: boolean }
  hub?: HubInfo | null // null unless the tutor was started as the menu-driven tutor
  learning?: LearningStatus | null // null unless the learn activity is running
  progress?: ProgressSummary | null
  speaking?: boolean // the tutor is talking right now
  settings?: { speech_speed: number; pace: string; tones: boolean; hearing_feedback: boolean; theme?: string | null }
  speech?: { text: string; started: number; seconds: number } | null // the line being spoken: when it started and how long it lasts
  heard?: { n: number; text: string; matched: boolean; reason: string | null } | null // the last thing the microphone made out (n counts up); null if it is not listening
  muted?: boolean // the learner turned the microphone off with the mute key
  // something said that the website does: a page ("my page", "sign out", "go back", "home"), scrolling, or pressing a control by its name
  nav?: {
    n: number
    to: 'login' | 'account' | 'signout' | 'click' | 'back' | 'home' | 'scroll-down' | 'scroll-up' | 'top' | null
    target?: string | null
  } | null
}

const post = (path: string, body: unknown) =>
  fetch(`${TUTOR_API}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

/** Say a voice command to the tutor without speaking: "next page", "repeat", "hint", "stop", ... */
export const sendCommand = (command: string) => post('/api/command', { command })

/** How much braille a new learner knows (asked before their first lesson): the lessons start from there. */
export type Level = 'new' | 'some' | 'alphabet' | 'numbers'
export const setLevel = (level: Level) => post('/api/level', { level })

/** Tell the tutor where the fingertip is on the video, as fractions (0-1) of its width and height. Stands in for the
 *  camera's own tracking for 10 s, then the camera has it back; send it again to keep it. */
export const sendFinger = (u: number, v: number) => post('/api/finger', { u, v })

/** Stop standing in for the fingertip: give the camera's own tracking back straight away. */
export const clearFinger = () => post('/api/finger', { clear: true })

export const speak = (text: string) => {
  try {
    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text))
  } catch {
    // no speech available in this browser: the same text is on screen
  }
}

/** The learner's whole progress record, as the tutor keeps it (opaque here: it is saved to and merged from the learner's account). */
export async function getProgress(): Promise<unknown> {
  const res = await fetch(`${TUTOR_API}/api/progress`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`progress: ${res.status}`)
  return (await res.json()).data
}

/** Fold a saved copy of the progress (e.g. from the learner's account) into the tutor's; merging the same copy twice changes nothing. */
export async function mergeProgress(data: unknown): Promise<void> {
  const res = await post('/api/progress', { data })
  if (!res.ok) throw new Error(`progress: ${res.status}`)
}

async function ok(res: Response, what: string): Promise<void> {
  if (!res.ok) throw new Error(`${what}: ${res.status}`)
}

/** Tell the tutor who is using it. A "google" learner's progress is kept (under `profile`, their account id); a "guest" keeps nothing. */
export const setSession = async (who: { kind: 'google' | 'guest'; name: string; profile?: string }) =>
  ok(await post('/api/session', who), 'session')

/** Nobody is signed in on the website: the tutor forgets whoever it had (quietly: no "signed out" line). */
export const forgetSession = async () => ok(await post('/api/session', { kind: 'none' }), 'session')

/** Choose what to do: learn, read, quiz, or menu to go back to the choice. */
export const setMode = async (mode: HubMode) => ok(await post('/api/mode', { mode }), 'mode')

/** Ask the tutor to say one of its prepared lines (only those it knows, see PROMPTS in tutor.py), with a first name where the line has one.
 * `cut`: stop whatever it is still saying first (the person answered over it). */
export const sayPrompt = async (name: string, who?: string, cut = false) =>
  ok(await post('/api/prompt', { name, ...(who === undefined ? {} : { who }), ...(cut ? { cut: true } : {}) }), 'prompt')

/** Tell the tutor which colours this page shows ("light", "dark", "contrast-light", "contrast"): the phone's page takes the same ones.
 * Nothing happens if the tutor is not running (the page on its own does not need it). */
export const shareTheme = (theme: string | undefined) => {
  if (theme) void post('/api/settings', { theme }).catch(() => {})
}

/** Show the backup link in the QR code (a tunnel that works on any network) instead of the local one. */
export const switchToBackupLink = async () => ok(await post('/api/phone/backup', {}), 'backup')

/** Tell the tutor which page is open and the names of the controls on it, so Braillo knows where the learner is and any of them can be
 * used by voice (see VoiceControl). Nothing happens if the tutor is not running. */
export const sharePage = (page: string, controls: string[], arrived = false) =>
  void post('/api/page', { page, controls, ...(arrived ? { arrived: true } : {}) }).catch(() => {})


/** The mute key: turn the tutor's microphone off (nothing is sent or acted on) or back on. */
export const setMic = async (on: boolean) => ok(await post('/api/mic', { on }), 'mic')

/** The sign-in page is (open) or is no longer talking with the user: while it is, the tutor does not act on voice commands or say "I did not catch that". */
export const setDialogue = async (open: boolean) => ok(await post('/api/dialogue', { open }), 'dialogue')

/** The tutor's state right now (one request). */
export async function fetchState(): Promise<TutorState> {
  const res = await fetch(`${TUTOR_API}/api/state`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`state: ${res.status}`)
  return res.json()
}
