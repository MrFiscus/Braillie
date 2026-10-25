import { useEffect, useRef } from 'react'
import type { Reach } from '../tutor/useTutorState.ts'
import { sayPrompt, setDialogue, type TutorState } from '../tutor/tutorApi.ts'
import { tutorCanTalk } from './channels.ts'
import { parseAccount, type AccountAsk } from './accountVoice.ts'
import { useVoiceTab } from './useVoiceTab.ts'

const MAX_MISSES = 2 // "Sorry, I didn't catch that" this many times in a row at most, then it waits quietly for something it knows

interface Args {
  reach: Reach
  state: TutorState | null
  name: string | null // who is here (null until known)
  act: (ask: Exclude<AccountAsk, 'none' | 'letters' | 'repeat' | 'help'>) => void // home, learn, read, quiz, sign out: the page does these
}

/**
 * Braillo on your page: once the tutor can talk, he reads out how you are getting on (the tutor fills in the numbers from its own
 * record), then listens. "Letters", "repeat" and "help" he answers himself; the rest the page does (`act`). While the page is open the
 * tutor is told it is in a conversation (as on the sign-in page), so it neither runs its own commands nor says "I did not catch that".
 */
export function useAccountVoice({ reach, state, name, act }: Args): void {
  const voiceTab = useVoiceTab() // (only the tab in use talks: two would cut each other off)
  const talks = reach === 'yes' && tutorCanTalk(state) && voiceTab
  const greeted = useRef(false)
  const lastHeard = useRef<number | null>(null)
  const misses = useRef(0)
  const latest = useRef({ name, act })
  useEffect(() => {
    latest.current = { name, act }
  })

  // greet once (a short wait first, so a page that is opened twice in development, or left again at once, does not speak twice)
  useEffect(() => {
    if (!talks || !name || greeted.current) return
    const t = window.setTimeout(() => {
      greeted.current = true
      void sayPrompt('account', name, true).catch(() => {})
    }, 400)
    return () => window.clearTimeout(t)
  }, [talks, name])

  // in a conversation while the page is open (kept open by a ping; closed when the page goes)
  useEffect(() => {
    if (reach !== 'yes' || !voiceTab) return
    void setDialogue(true).catch(() => {})
    const ping = window.setInterval(() => void setDialogue(true).catch(() => {}), 10000)
    return () => {
      window.clearInterval(ping)
      void setDialogue(false).catch(() => {})
    }
  }, [reach, voiceTab])

  const heard = state?.heard ?? null
  useEffect(() => {
    if (!heard) return
    if (lastHeard.current === null || heard.n < lastHeard.current) {
      lastHeard.current = heard.n // what was heard before this page opened is not for it
      return
    }
    if (heard.n === lastHeard.current) return
    lastHeard.current = heard.n
    if (!heard.text || !greeted.current || !voiceTab) return
    const ask = parseAccount(heard.text)
    const who = latest.current.name ?? undefined
    if (ask === 'none') {
      // a word or two caught over Braillo's own voice is most likely Braillo, misheard: let it go quietly
      if (heard.reason === 'heard over speech' || ++misses.current > MAX_MISSES) return
      void sayPrompt('account_retry', undefined, true).catch(() => {})
      return
    }
    misses.current = 0
    if (ask === 'letters') void sayPrompt('account_letters', undefined, true).catch(() => {})
    else if (ask === 'repeat') void sayPrompt('account', who, true).catch(() => {})
    else if (ask === 'help') void sayPrompt('account_help', undefined, true).catch(() => {})
    else latest.current.act(ask)
  }, [heard, voiceTab])
}
