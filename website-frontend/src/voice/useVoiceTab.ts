import { useEffect, useState } from 'react'

// Only one tab may talk with Braillo. With the website open in two tabs (or a page left open from before), each would run its own
// conversation with the one tutor, and they would cut each other's lines off midway. The tab in use leads: the one in front with the
// focus, or the one last clicked or typed in. A tab out of sight never leads; and if the leading tab goes quiet (closed, or put away)
// for a few seconds, a tab that is in sight takes over by itself, so there is always one.

const KEY = 'braillie.voiceTab'
const ME = Math.random().toString(36).slice(2)
const BEAT_MS = 1000
const STALE_MS = 3500

interface Lead {
  id: string
  at: number // when the leading tab last said it is still there
}

const read = (): Lead | null => {
  try {
    const raw = window.localStorage.getItem(KEY)
    const lead = raw ? (JSON.parse(raw) as Lead) : null
    return lead && typeof lead.id === 'string' && typeof lead.at === 'number' ? lead : null
  } catch {
    return null
  }
}
const write = () => {
  try {
    window.localStorage.setItem(KEY, JSON.stringify({ id: ME, at: Date.now() }))
  } catch {
    // no storage here: this tab simply leads
  }
}
const visible = () => document.visibilityState === 'visible'

/** Look, and take the lead if this tab should have it. Returns whether it leads. */
function check(): boolean {
  if (!visible()) return false
  const lead = read()
  const mine = !lead || lead.id === ME
  if (mine || document.hasFocus() || Date.now() - lead.at > STALE_MS) {
    write() // (in front with the focus, or nobody else is there: this one leads, and says so again)
    return true
  }
  return false
}

/** Is this the tab that talks with Braillo? (It changes as the person moves between tabs.) */
export function useVoiceTab(): boolean {
  const [leads, setLeads] = useState(check)

  useEffect(() => {
    const update = () => setLeads(check())
    const used = () => {
      if (visible()) write() // clicked or typed in: this is the tab in use
      update()
    }
    const beat = window.setInterval(update, BEAT_MS)
    window.addEventListener('focus', update)
    window.addEventListener('pointerdown', used)
    window.addEventListener('keydown', used)
    document.addEventListener('visibilitychange', update)
    window.addEventListener('storage', update) // (another tab took over)
    return () => {
      window.clearInterval(beat)
      window.removeEventListener('focus', update)
      window.removeEventListener('pointerdown', used)
      window.removeEventListener('keydown', used)
      document.removeEventListener('visibilitychange', update)
      window.removeEventListener('storage', update)
    }
  }, [])

  return leads
}
