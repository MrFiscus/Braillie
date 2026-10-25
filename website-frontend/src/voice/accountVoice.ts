// What people say on their page (the profile), kept free of the browser and the network so it can be tested. The tutor speaks the
// lines (PROMPTS "account..." in braille_tutor/tutor.py, with the learner's numbers filled in from its own record); this only works out
// what was asked for.
//
//   Braillo: "This is your page, Sam. You know 3 of 26 letters... Say continue to go home, letters to hear your alphabet, or sign out."
//   You:     "which letters do I know"   -> letters
//   You:     "let's carry on"            -> home

export type AccountAsk = 'home' | 'letters' | 'learn' | 'read' | 'quiz' | 'signout' | 'repeat' | 'help' | 'none'

const clean = (text: string) =>
  text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}' -]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()

/** What was asked for. Checked most particular first: "read my letters" is about the letters, not the Read activity. */
export function parseAccount(text: string): AccountAsk {
  const t = clean(text)
  if (!t) return 'none'
  if (/\b((sign|log)[ -]?(me )?out|signing out)\b/.test(t)) return 'signout'
  if (/\b(letters?|alphabet|progress|how am i doing|how'?m i doing|what do i know|which ones)\b/.test(t)) return 'letters'
  if (/\b(help|what can i say|options)\b/.test(t)) return 'help'
  if (/\b(repeat|again|pardon|say that)\b/.test(t)) return 'repeat'
  if (/\b(learn|lesson|teach)\b/.test(t)) return 'learn'
  if (/\b(read|reading)\b/.test(t)) return 'read'
  if (/\b(quiz|test me)\b/.test(t)) return 'quiz'
  if (/\b(continue|carry on|home|keep going|go on|next|done|okay|ok|menu)\b/.test(t)) return 'home' // ("go back" is the page before, for every page)
  return 'none'
}
