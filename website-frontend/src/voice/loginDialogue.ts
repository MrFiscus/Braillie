// The spoken conversation on the sign-in page, kept free of the browser and the network so it can be tested: what people say is turned
// into a choice, a name or a yes/no here, and `runLoginDialogue` walks the conversation using a `Channel` (the tutor's voice, or the
// browser's) to speak and to listen.
//
//   Braillie: "Welcome ... say Google to sign in, or guest to carry on without an account."
//   You:      "guest"
//   Braillie: "Okay, without an account ... What is your first name? Or say skip."
//   You:      "my name is Sam"
//   Braillie: "Is your name Sam? Say yes, or say your name again."
//   You:      "yes"                      -> continues as a guest called Sam

/** Everything Braillie can say on this page. The tutor has the same lines (PROMPTS in braille_tutor/tutor.py); the browser voice uses these. */
export const PROMPT_TEXT = {
  welcome: () =>
    'Welcome to Braillie. On this page you can sign in with Google, or continue without an account. To sign in, say Google. To carry on without an account, say guest. Or use the tab key to move between the options. If you sign in, your progress is remembered. If you continue without an account, it is not.',
  welcome_back: (who: string) => `Welcome back, ${who}. Say continue to carry on as ${who}, or say switch to choose another way.`,
  retry_choice: () => 'Sorry, I did not catch that. Say Google to sign in, or say guest to continue without an account.',
  retry_back: () => 'Sorry, I did not catch that. Say continue, or say switch.',
  ask_name: () => 'Okay, without an account. Your progress will not be saved. What is your first name? Or say skip.',
  confirm_name: (who: string) => `Is your name ${who}? Say yes, or say your name again.`,
  again_name: () => 'Okay. Please say your first name again, or say skip.',
  retry_name: () => 'Sorry, I did not catch that. Please say your first name, or say skip.',
  go_guest: (who: string) => `Nice to meet you, ${who}. Let us connect your phone.`,
  go_guest_anon: () => 'Okay. Let us connect your phone.',
  go_google: () => 'Opening Google. From here, use your keyboard or screen reader on the Google page.',
  give_up: () => 'I will stop listening now. You can use the buttons on the screen: press tab to move between them.',
} as const
export type PromptName = keyof typeof PROMPT_TEXT

// ---- understanding what was said ----------------------------------------------------------------------------------------------------
const clean = (text: string) =>
  text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}' -]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()

export type Choice = 'google' | 'guest' | 'repeat' | 'none'

/** "Google" or "guest"? Speech recognition spells Google many ways, and people say more than one word ("sign in with Google please"). */
export function parseChoice(text: string): Choice {
  const t = clean(text)
  if (/\b(guest|without|no account|skip|anonymous|don'?t have|do not have|carry on|continue)\b/.test(t)) return 'guest'
  if (/\bgo+g+(le|el|ol|al)s?\b|\bgmail\b|\bsign ?in\b|\blog ?in\b/.test(t)) return 'google'
  if (/\b(repeat|again|what|help|pardon)\b/.test(t)) return 'repeat'
  return 'none'
}

export type Back = 'continue' | 'switch' | 'repeat' | 'none'

/** For someone who is already known: carry on, or use another account. */
export function parseBack(text: string): Back {
  const t = clean(text)
  if (/\b(switch|change|different|another|someone else|not me|sign out|log out|no)\b/.test(t)) return 'switch'
  if (/\b(continue|carry on|yes|yeah|yep|go on|go ahead|ok|okay|sure|that'?s me|that is me|it'?s me|it is me|correct)\b/.test(t)) return 'continue'
  if (/\b(repeat|again|what|help|pardon)\b/.test(t)) return 'repeat'
  return 'none'
}

export type YesNo = 'yes' | 'no' | 'other'

export function parseYesNo(text: string): YesNo {
  const t = clean(text)
  if (/^(yes|yeah|yep|yup|correct|right|that'?s right|that is right|sure|ok|okay|affirmative|exactly)\b/.test(t)) return 'yes'
  if (/^(no|nope|nah|wrong|incorrect|not)\b/.test(t)) return 'no'
  return 'other'
}

const FILLER = /^(?:(?:hi|hello|hey|well|um|uh|er|so|okay|ok)\s+)*(?:(?:my (?:first )?names?(?: is| s)?|the name is|i am|i'?m|im|it'?s|its|this is|call me|you can call me|they call me)\s+)*/

// Words that are answers or chatter, never someone's name (speech recognition happily returns them when it is unsure).
const NOT_NAMES = new Set(['yes', 'yeah', 'yep', 'no', 'nope', 'nah', 'ok', 'okay', 'sure', 'hello', 'hi', 'hey', 'google', 'guest', 'sorry', 'what', 'help', 'repeat',
  'again', 'um', 'uh', 'er', 'hmm', 'the', 'a', 'and', 'thanks', 'thank', 'please', 'continue', 'switch', 'stop'])

/** A first name from "Sam", "my name is Sam", "I'm Sam, please". "skip" (or "no name") means they would rather not say. null: nothing usable. */
export function parseName(text: string): string | 'skip' | null {
  const t = clean(text)
  if (!t) return null
  if (/^(skip|pass|none|nothing|no name|anonymous|i'?d rather not|rather not|never ?mind)\b/.test(t)) return 'skip'
  const rest = t.replace(FILLER, '').replace(/\b(please|thanks|thank you)\b.*$/, '').trim()
  const word = rest.split(' ')[0]?.replace(/^[-']+|[-']+$/g, '') ?? ''
  if (!/^\p{L}[\p{L}'-]*$/u.test(word) || word.length > 24 || NOT_NAMES.has(word)) return null
  return word.replace(/(^|[-'])(\p{L})/gu, (_m, edge: string, letter: string) => edge + letter.toLocaleUpperCase())
}

// ---- the conversation ---------------------------------------------------------------------------------------------------------------

/** A voice to speak with and a mic to listen with. `say` resolves when it has finished talking; `listen` gives what was heard, or null if nothing was. */
export interface Channel {
  say(prompt: PromptName, who?: string): Promise<void>
  listen(signal: AbortSignal): Promise<string | null>
}

export interface Actions {
  google(): void | Promise<void>
  guest(name: string): void
  carryOn(): void
  switchAccount(): void | Promise<void>
}

export type Phase = 'speaking' | 'listening' | 'idle'
export interface Status {
  phase: Phase
  heard?: string // the last thing understood, shown on screen
  hint?: string // what can be said now
}

export type Outcome = 'google' | 'guest' | 'carried-on' | 'gave-up' | 'stopped'

export interface DialogueOptions {
  channel: Channel
  actions: Actions
  returning: string | null // the name of someone already known, or null
  onStatus: (s: Status) => void
  signal: AbortSignal
  maxMisses?: number // silences or things not understood, in a row, before Braillie stops asking
}

const HINT = { choice: 'Say “Google” or “guest”', back: 'Say “continue” or “switch”', name: 'Say your first name, or “skip”', confirm: 'Say “yes”, or say your name again' }

export async function runLoginDialogue(o: DialogueOptions): Promise<Outcome> {
  const { channel, actions, signal, onStatus } = o
  const maxMisses = o.maxMisses ?? 3
  type Step = 'choice' | 'back' | 'name' | 'confirm'
  let step: Step = o.returning ? 'back' : 'choice'
  let candidate = ''
  let heard: string | undefined
  let misses = 0

  const say = async (prompt: PromptName, who?: string) => {
    onStatus({ phase: 'speaking', heard, hint: HINT[step] })
    await channel.say(prompt, who)
  }
  const done = (outcome: Outcome): Outcome => {
    onStatus({ phase: 'idle', heard })
    return outcome
  }
  const miss = async (retry: PromptName): Promise<Outcome | null> => {
    if (++misses >= maxMisses) {
      await say('give_up')
      return done('gave-up')
    }
    await say(retry)
    return null
  }

  await say(o.returning ? 'welcome_back' : 'welcome', o.returning ?? undefined)
  while (!signal.aborted) {
    onStatus({ phase: 'listening', heard, hint: HINT[step] })
    const text = await channel.listen(signal)
    if (signal.aborted) break
    if (text === null || !clean(text)) {
      const end = await miss(step === 'choice' ? 'retry_choice' : step === 'back' ? 'retry_back' : 'retry_name')
      if (end) return end
      continue
    }
    heard = text.trim()
    onStatus({ phase: 'speaking', heard, hint: HINT[step] })

    if (step === 'choice') {
      const c = parseChoice(text)
      if (c === 'google') {
        await say('go_google')
        await actions.google()
        return done('google')
      }
      if (c === 'guest') {
        step = 'name'
        misses = 0
        await say('ask_name')
      } else if (c === 'repeat') await say('welcome')
      else {
        const end = await miss('retry_choice')
        if (end) return end
      }
    } else if (step === 'back') {
      const b = parseBack(text)
      if (b === 'continue') {
        actions.carryOn()
        return done('carried-on')
      }
      if (b === 'switch') {
        await actions.switchAccount()
        step = 'choice'
        misses = 0
        await say('welcome')
      } else if (b === 'repeat') await say('welcome_back', o.returning ?? undefined)
      else {
        const end = await miss('retry_back')
        if (end) return end
      }
    } else if (step === 'name' || step === 'confirm') {
      const yn = step === 'confirm' ? parseYesNo(text) : 'other'
      if (yn === 'yes') {
        await say('go_guest', candidate)
        actions.guest(candidate)
        return done('guest')
      }
      if (yn === 'no') {
        step = 'name'
        misses = 0
        await say('again_name')
        continue
      }
      const name = parseName(text)
      if (name === 'skip') {
        await say('go_guest_anon')
        actions.guest('')
        return done('guest')
      }
      if (name) {
        candidate = name
        step = 'confirm'
        misses = 0
        await say('confirm_name', name)
      } else {
        const end = await miss('retry_name')
        if (end) return end
      }
    }
  }
  return done('stopped')
}
