// Talks to the braille tutor (braille_tutor/tutor_server.py, started on the laptop). The API is documented in braille_tutor/API.md.
// Override the address with VITE_TUTOR_API (e.g. in .env.local) if the tutor is not on this computer's port 8000.
export const TUTOR_API: string = import.meta.env.VITE_TUTOR_API ?? 'http://127.0.0.1:8000'

export interface PhoneInfo {
  connected: boolean
  address: string
  code: string
  qr: string // path of the QR code image, relative to TUTOR_API
  instructions: string
  diagnosis?: string | null // why no video has arrived yet, in plain words
  sound?: boolean
  mic?: boolean
}

export type LearningPhase = 'idle' | 'await_sheet' | 'teach' | 'practice' | 'review' | 'recap' | 'explore' | 'done'

/** Where the learner is in the guided lessons (only when the tutor runs with --mode learn). */
export interface LearningStatus {
  phase: LearningPhase
  lesson: { id: string; title: string; index: number; of: number; sheet: string } | null
  target: string | null // the letter being asked for
  target_dots: number[]
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
  tutor: { mode: string; state: string; prompt: string }
  finger: { page_mm: [number, number] | null; cell: { letter: string | null; label: string | null; name: string | null; dots: number[] } | null }
  said: { t: number; kind: string; text: string }[]
  reading?: { locked: number; total: number; between_pages: boolean }
  learning?: LearningStatus | null // null unless --mode learn
  progress?: ProgressSummary | null
}

const post = (path: string, body: unknown) =>
  fetch(`${TUTOR_API}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

/** Say a voice command to the tutor without speaking: "next page", "repeat", "hint", "stop", ... */
export const sendCommand = (command: string) => post('/api/command', { command })

/** Tell the tutor where the fingertip is on the video, as fractions (0-1) of its width and height. */
export const sendFinger = (u: number, v: number) => post('/api/finger', { u, v })

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
