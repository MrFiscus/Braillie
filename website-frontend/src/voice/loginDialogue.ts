// The spoken conversation on the sign-in page, kept free of the browser and the network so it can be tested: what people say is turned
// into a choice, a name or a yes/no here, and `runLoginDialogue` walks the conversation using a `Channel` (the tutor's voice, or the
// browser's) to speak and to listen.
//
//   Braillie: "Welcome ... say Google to sign in, or guest to carry on without an account."
//   You:      "guest"
//   Braillie: "Hi, guest!"   -> continues as a guest (the next page, connecting the phone, says what to do there)

/** Everything Braillie can say on this page. The tutor has the same lines (PROMPTS in braille_tutor/tutor.py); the browser voice uses these.
 * The lines heard at every start have a few versions saying the same thing in other words, so it does not sound like one script read
 * again and again: PROMPT_TEXT gives the version that will be said next (the same each time it is asked, so the words on screen match),
 * and `nextVersion` moves on to another one once it has been said. */
const spelled = (code: string) => (code ? code.split('').join(' ') : 'shown on the screen')
const VERSIONS = {
  welcome: [
    () =>
      "Hi, I'm Braillo, and I'll be teaching you braille. Would you like to sign in with Google, so I can remember your progress, or carry on as a guest? Just say Google, or guest.",
    () =>
      "Welcome to Braillie! I'm Braillo. If you sign in with Google, I'll keep track of how you're doing. If you'd rather not, you can come in as a guest. Say Google, or guest.",
    () =>
      "Hello, and welcome to Braillie. I'm Braillo, your braille tutor. To save your progress, say Google and sign in. To just try it out, say guest. You can also use the tab key to move between the options.",
    () =>
      "Hi there, welcome to Braillie. You can sign in with Google to keep your progress, or come in as a guest, without an account. Say Google, or say guest.",
    () =>
      "Welcome! I'm Braillo. Before we begin, would you like to sign in? Say Google to save your progress, or guest to jump straight in.",
    () =>
      "Hello! This is Braillie, and I'm Braillo. Say Google if you'd like me to remember your progress, or say guest to start without an account.",
    () =>
      "Hi, I'm Braillo. Welcome to Braillie. There are two ways in: say Google to sign in and save your progress, or say guest to start right away.",
    () =>
      "Welcome to Braillie. I'm Braillo, and I'm here to help you learn braille by touch. Say Google to sign in, or guest to carry on without an account.",
    () =>
      "Hello, welcome to Braillie! Would you like to sign in with Google, so your progress is saved? Say Google, or say guest to come in as a guest. The tab key works too.",
    () =>
      "Hi! Braillo here. Welcome to Braillie. Sign in with Google and I'll remember where you got to, or say guest to start fresh, without an account.",
  ],
  welcome_back: [
    (who: string) =>
      `Welcome back, ${who}! Say continue to pick up where you left off, or switch if this isn't you.`,
    (who: string) =>
      `Hi again, ${who}. Shall we carry on? Say continue, or say switch to use a different account.`,
    (who: string) =>
      `Good to see you again, ${who}. Say continue to keep going, or switch if you're someone else.`,
    (who: string) =>
      `Hello again, ${who}! Ready to carry on? Just say continue. Or say switch to sign in another way.`,
    (who: string) =>
      `Nice to have you back, ${who}. Say continue, and we'll pick up where we left off. Or say switch.`,
    (who: string) =>
      `${who}, welcome back! Say continue to carry on, or switch if this isn't you.`,
    (who: string) =>
      `Hey, ${who}, good to hear from you again. Say continue when you're ready, or switch to change accounts.`,
    (who: string) =>
      `Welcome back! Is that you, ${who}? Say continue if it is, or switch if it's someone else.`,
    (who: string) =>
      `Hi, ${who}, back for more? Say continue to pick up where you left off, or say switch.`,
    (who: string) =>
      `There you are, ${who}! Say continue to keep going, or switch to sign in another way.`,
  ],
  retry_choice: [() => "Sorry, I didn't catch that. Say Google to sign in, or guest to carry on without an account."],
  retry_back: [() => "Sorry, I didn't catch that. Say continue, or switch."],
  ask_name: [
    () =>
      "Okay, no account. Just so you know, your progress won't be saved this time. What's your first name? Or say skip.",
    () =>
      "Sure, we'll go without an account, so nothing gets saved. What should I call you? You can also say skip.",
    () =>
      "Alright, as a guest, then. Your progress won't be kept, but that's fine. What's your first name? Or just say skip.",
    () =>
      "No account, no problem. Remember, nothing is saved this way. What's your first name? Say skip if you'd rather not.",
    () =>
      "Great, let's start as a guest. Your progress won't be saved. What would you like me to call you? Or say skip.",
    () =>
      "Okay, guest it is. Nothing will be saved, just so you know. What's your first name? You can say skip, too.",
    () =>
      "Got it, no account. Your progress won't be kept this time. What's your first name? Or say skip.",
    () =>
      "Fine by me. Without an account, I won't keep your progress. Could you tell me your first name? Or say skip.",
    () =>
      "Okay, let's skip the account. Your progress won't be saved. What's your first name? Or say skip.",
    () =>
      "Sure thing. As a guest, nothing gets saved. What's your first name? If you'd rather not say, just say skip.",
  ],
  confirm_name: [(who: string) => `Is your name ${who}? Say yes, or tell me your name again.`],
  again_name: [() => "Okay. What's your first name, then? Or say skip."],
  retry_name: [() => "Sorry, I didn't catch that. What's your first name? Or say skip."],
  go_guest: [
    (who: string) =>
      `Nice to meet you, ${who}! Now let's connect your phone.`,
    (who: string) =>
      `Lovely to meet you, ${who}. Next, let's get your phone connected.`,
    (who: string) =>
      `Hi, ${who}! Let's connect your phone next.`,
    (who: string) =>
      `Great to meet you, ${who}. Now for your phone.`,
    (who: string) =>
      `Hello, ${who}! Next up, your phone.`,
    (who: string) =>
      `Welcome, ${who}. Let's get your phone set up.`,
    (who: string) =>
      `Nice one, ${who}. Now let's link your phone.`,
    (who: string) =>
      `Pleased to meet you, ${who}. Let's set up your phone.`,
    (who: string) =>
      `Hi there, ${who}! Time to connect your phone.`,
    (who: string) =>
      `Good to meet you, ${who}. Let's get your phone ready.`,
  ],
  go_guest_anon: [
    () => "Hi, guest!",
    () => "Hello, guest!",
    () => "Welcome, guest!",
    () => "Hi there, guest!",
    () => "Welcome in, guest!",
    () => "Hello there, guest!",
    () => "Hi, guest, good to have you!",
    () => "Welcome aboard, guest!",
    () => "Hey there, guest!",
    () => "Hi, guest! Glad you're here.",
  ],
  go_google: [() => "Opening Google now. From here, use your keyboard or screen reader on the Google page."],
  give_up: [() => "I'll stop listening for now. You can use the buttons on the screen instead: press tab to move between them."],
  // (all the steps, once: scan, the link, Start on the phone, where to hold it. Troubleshooting only when they ask)
  connect_phone: [
    () => `Now scan the code on the screen with your phone's camera, and tap the link. Then tap the big Start button in the middle of the phone's screen, and hold the phone about thirty centimetres above your sheet.`,
    () => `Next, your phone. Scan the code on the screen and tap the link that appears. Tap Start, the big button in the middle of its screen, then hold the phone about thirty centimetres above the sheet.`,
    () => `Point your phone's camera at the code on the screen and tap the link. When the page opens, tap the big Start button in the middle, and hold the phone about thirty centimetres above your sheet.`,
    () => `Time to connect your phone. Scan the code on the screen, tap the link, then tap the Start button in the middle of the phone's screen. Hold the phone about thirty centimetres above the sheet.`,
    () => `Now your phone: scan the code on the screen and tap the link. Tap the big Start button in the middle of the phone, then hold it about thirty centimetres above your sheet, camera facing down.`,
    () => `Let's link your phone. Scan the code on the screen with its camera and tap the link. Then tap Start in the middle of the phone's screen, and hold the phone about thirty centimetres above the sheet.`,
    () => `Scan the code on the screen with your phone and tap the link that comes up. Tap the big Start button in the middle of its screen, then hold the phone about thirty centimetres over your sheet.`,
    () => `Next, scan the code on the screen with your phone's camera and open the link. Tap the Start button in the middle of the phone's screen, and hold the phone about thirty centimetres above the sheet.`,
    () => `Okay, your phone. Point its camera at the code on the screen and tap the link. Tap the big Start button in the middle of the screen, then hold the phone about thirty centimetres above your sheet.`,
    () => `Now let's connect your phone. Scan the code on the screen and tap the link. Then tap Start, in the middle of the phone's screen, and hold the phone about thirty centimetres above the sheet.`,
  ],
  // (the phone dropped out for good, mid-way: the code is shown again, see usePhoneWatch.ts)
  reconnect_phone: [() => `I've lost your phone. Scan the code on the screen with its camera again, tap Start, and we'll carry on where we were.`],
  // (only when they say it is not working: the way in without scanning)
  connect_help: [
    (code: string) =>
      `Let's sort it out. If the code won't scan, open the address on the screen in your phone's browser, and type the code ${spelled(code)}. The phone needs the internet, on Wi-Fi or mobile data. When the page opens, tap Start, and allow the camera.`,
  ],
} as const satisfies Record<string, readonly ((w: string) => string)[]>

const planned: Partial<Record<keyof typeof VERSIONS, number>> = {}
const pick = (n: number, not?: number) => {
  const choices = [...Array(n).keys()].filter((i) => i !== not)
  return choices.length ? choices[Math.floor(Math.random() * choices.length)] : 0
}
const plannedIndex = (name: keyof typeof VERSIONS) => (planned[name] ??= pick(VERSIONS[name].length))

/** The line to say for each name: the version that will be said next (see nextVersion). */
export const PROMPT_TEXT = Object.fromEntries(
  Object.keys(VERSIONS).map((name) => {
    const key = name as keyof typeof VERSIONS
    return [key, (w: string) => (VERSIONS[key][plannedIndex(key)] as (w: string) => string)(w)]
  }),
) as { [K in keyof typeof VERSIONS]: (w: string) => string }

/** This line has been said: next time, another version of it (never the same twice running). */
export function nextVersion(name: keyof typeof VERSIONS): void {
  planned[name] = pick(VERSIONS[name].length, plannedIndex(name))
}

/** Every version of every line (for checking them against the tutor's). */
export const ALL_VERSIONS = VERSIONS
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
  if (/\b(switch|change|different|another|someone else|somebody else|not me|sign out|log out|no|nope|new account|start over)\b/.test(t)) return 'switch'
  if (/\b(continue|carry on|keep going|yes|yeah|yep|yup|go on|go ahead|go|let'?s go|let'?s start|start|proceed|next|skip|ok|okay|sure|fine|alright|all right|right|good|that'?s me|that is me|it'?s me|it is me|me|correct)\b/.test(t)) return 'continue'
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
  'again', 'um', 'uh', 'er', 'hmm', 'the', 'a', 'and', 'thanks', 'thank', 'please', 'continue', 'switch', 'stop', 'skip', 'i', 'me', 'my', 'it',
  'is', 'just', 'so', 'well', 'oh', 'wait', 'quiet', 'can', 'could', 'how', 'why', 'who', 'where', 'when', 'do', 'don\'t', 'not', 'that', 'this',
  'there', 'here', 'you', 'we', 'they', 'go', 'let\'s', 'start', 'next', 'okay', 'fine', 'cool', 'great', 'good', 'right', 'name'])

/** A first name from "Sam", "my name is Sam", "I'm Sam, please". "skip" (or "no name") means they would rather not say. null: nothing usable. */
export function parseName(text: string): string | 'skip' | null {
  const t = clean(text)
  if (!t) return null
  if (/^(skip|pass|none|nothing|no name|no thanks|no thank you|anonymous|i'?d rather not|rather not|never ?mind|not telling|i don'?t want|i do not want|doesn'?t matter|it doesn'?t matter|whatever)\b/.test(t)) return 'skip'
  const rest = t.replace(FILLER, '').replace(/\b(please|thanks|thank you)\b.*$/, '').trim()
  const word = rest.split(' ')[0]?.replace(/^[-']+|[-']+$/g, '') ?? ''
  if (!/^\p{L}[\p{L}'-]*$/u.test(word) || word.length < 2 || word.length > 24 || NOT_NAMES.has(word)) return null
  return word.replace(/(^|[-'])(\p{L})/gu, (_m, edge: string, letter: string) => edge + letter.toLocaleUpperCase())
}

// ---- the conversation ---------------------------------------------------------------------------------------------------------------

/** A voice to speak with and a mic to listen with. `say` resolves when it has finished talking; `listen` gives what was heard, or null if nothing was. */
export interface Channel {
  say(prompt: PromptName, who?: string): Promise<void>
  listen(signal: AbortSignal): Promise<string | null>
  /** Was what `listen` last gave caught while Braillie was still talking? Then it may be Braillie's own voice, misheard: if it is not an
   * answer, it is let go quietly instead of "Sorry, I did not catch that". */
  heardOverSpeech?(): boolean
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
  saying?: string // while speaking: the words being said, shown on screen as they are spoken
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

export const GUEST_NAME = 'Guest' // what a guest is called (they are not asked for a name)

export async function runLoginDialogue(o: DialogueOptions): Promise<Outcome> {
  const { channel, actions, signal, onStatus } = o
  const maxMisses = o.maxMisses ?? 3
  type Step = 'choice' | 'back' | 'name' | 'confirm'
  let step: Step = o.returning ? 'back' : 'choice'
  let candidate = ''
  let heard: string | undefined
  let misses = 0

  const say = async (prompt: PromptName, who?: string) => {
    onStatus({ phase: 'speaking', heard, hint: HINT[step], saying: (PROMPT_TEXT[prompt] as (w: string) => string)(who ?? '') })
    await channel.say(prompt, who)
  }
  const done = (outcome: Outcome): Outcome => {
    onStatus({ phase: 'idle', heard })
    return outcome
  }
  let overSpeech = false // what was just heard came while Braillie was still talking
  const miss = async (retry: PromptName): Promise<Outcome | null> => {
    if (overSpeech) return null // probably Braillie's own voice, misheard: not worth a "sorry", and it keeps listening
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
    overSpeech = text !== null && !!channel.heardOverSpeech?.()
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
        // straight in, as "Guest": no name to ask for (a guest's name is not kept anyway)
        await say('go_guest_anon')
        actions.guest(GUEST_NAME)
        return done('guest')
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
