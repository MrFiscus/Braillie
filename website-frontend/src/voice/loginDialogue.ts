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

/** Everything Braillie can say on this page. The tutor has the same lines (PROMPTS in braille_tutor/tutor.py); the browser voice uses these.
 * The lines heard at every start have a few versions saying the same thing in other words, so it does not sound like one script read
 * again and again: PROMPT_TEXT gives the version that will be said next (the same each time it is asked, so the words on screen match),
 * and `nextVersion` moves on to another one once it has been said. */
const spelled = (code: string) => (code ? code.split('').join(' ') : 'shown on the screen')
const VERSIONS = {
  welcome: [
    () =>
      "Welcome to Braillie. On this page you can sign in with Google, or continue without an account. To sign in, say Google. To carry on without an account, say guest. Or use the tab key to move between the options. If you sign in, your progress is remembered. If you continue without an account, it is not.",
    () =>
      "Hi, and welcome to Braillie. You can sign in with Google, so your progress is remembered, or carry on as a guest, and nothing is saved. Say Google, or say guest. You can also press tab to move between the options.",
    () =>
      "Hello! I'm Braillo, and this is Braillie. To keep your progress, say Google to sign in. To try it without an account, say guest, but your progress won't be saved. Or press tab to move between the options.",
    () =>
      "Hi there, welcome to Braillie. There are two ways in. Say Google to sign in and keep your progress, or say guest to start straight away without saving anything. The tab key moves between the options too.",
    () =>
      "Welcome! I'm Braillo, your braille tutor. Would you like to sign in? Say Google, and I'll remember how you're getting on. Or say guest to carry on without an account, and nothing is kept. You can press tab to move between the options instead.",
    () =>
      "Hello, and welcome to Braillie. Say Google to sign in with your Google account, and your progress is saved. Say guest to carry on without one, and it is not. Or use tab to move between the options.",
    () =>
      "Good to have you here. This is Braillie, and I'm Braillo. To save your progress, sign in by saying Google. To just try it out, say guest, and nothing will be saved. Pressing tab moves between the options as well.",
    () =>
      "Welcome to Braillie! First, how would you like to start? Say Google to sign in, so I can remember your progress. Say guest to go without an account, and your progress won't be kept. Or press tab to move between the options.",
    () =>
      "Hi! Braillo here, welcome to Braillie. You can say Google to sign in and keep your progress from one visit to the next, or say guest to carry on without an account, which saves nothing. The tab key moves between the options too.",
    () =>
      "Hello, welcome to Braillie. Two choices: say Google to sign in, and your progress is remembered, or say guest to continue without an account, and it is forgotten when you leave. You can also use tab to move between them.",
  ],
  welcome_back: [
    (who: string) => `Welcome back, ${who}. Say continue to carry on as ${who}, or say switch to choose another way.`,
    (who: string) => `Hi again, ${who}! Say continue to pick up where you left off, or say switch to use another way in.`,
    (who: string) => `Good to see you, ${who}. Say continue to carry on, or say switch if this isn't you.`,
    (who: string) => `Hello again, ${who}. To carry on, say continue. To sign in another way, say switch.`,
    (who: string) => `Nice to have you back, ${who}! Say continue to keep going, or switch to choose a different way in.`,
    (who: string) => `${who}, welcome back. Say continue, and we'll carry on. Say switch if you'd like another way in.`,
    (who: string) => `Hey, ${who}, good to hear from you again. Say continue to carry on as ${who}, or say switch.`,
    (who: string) => `Welcome back! Is that you, ${who}? Say continue to carry on, or say switch to choose another way.`,
    (who: string) => `Hi, ${who}, back for more? Say continue to pick up where you left off, or say switch if it's someone else.`,
    (who: string) => `There you are, ${who}. Say continue to carry on, or say switch to sign in another way.`,
  ],
  retry_choice: [() => 'Sorry, I did not catch that. Say Google to sign in, or say guest to continue without an account.'],
  retry_back: [() => 'Sorry, I did not catch that. Say continue, or say switch.'],
  ask_name: [
    () =>
      "Okay, without an account. Your progress will not be saved. What is your first name? Or say skip.",
    () =>
      "Okay, no account. Just so you know, your progress won't be saved. What's your first name? You can also say skip.",
    () =>
      "Sure, let's go without an account. Nothing will be saved. What should I call you? Say your first name, or say skip.",
    () =>
      "Alright, as a guest, then. Your progress won't be kept. What's your first name? Or just say skip.",
    () =>
      "No account, no problem. Remember, nothing is saved this way. Tell me your first name, or say skip.",
    () =>
      "Great, we'll carry on without an account, so your progress won't be saved. What's your first name? Say skip if you'd rather not.",
    () =>
      "Okay, guest it is. Your progress won't be saved. What would you like me to call you? Say your first name, or say skip.",
    () =>
      "Got it, no account. Just a heads up, nothing gets saved. What's your first name? Or say skip.",
    () =>
      "Fine by me. Without an account, your progress isn't kept. Could you tell me your first name? Or say skip.",
    () =>
      "Okay, let's skip the account. Your progress won't be saved. What's your first name? You can say skip, too.",
  ],
  confirm_name: [(who: string) => `Is your name ${who}? Say yes, or say your name again.`],
  again_name: [() => 'Okay. Please say your first name again, or say skip.'],
  retry_name: [() => 'Sorry, I did not catch that. Please say your first name, or say skip.'],
  go_guest: [
    (who: string) => `Nice to meet you, ${who}. Let us connect your phone.`,
    (who: string) => `Lovely to meet you, ${who}. Next, let's connect your phone.`,
    (who: string) => `Hi, ${who}! Now let's get your phone connected.`,
    (who: string) => `Great to meet you, ${who}. Now, your phone.`,
    (who: string) => `Hello, ${who}! Let's connect your phone next.`,
    (who: string) => `Welcome, ${who}. Next up, connecting your phone.`,
    (who: string) => `Nice one, ${who}. Now let's link your phone.`,
    (who: string) => `Pleased to meet you, ${who}. Let's set up your phone.`,
    (who: string) => `Hi there, ${who}! Time to connect your phone.`,
    (who: string) => `Good to meet you, ${who}. Let's get your phone ready.`,
  ],
  go_guest_anon: [
    () =>
      "Okay. Let us connect your phone.",
    () =>
      "No problem. Next, let's connect your phone.",
    () =>
      "Alright. Now let's get your phone connected.",
    () =>
      "Sure thing. Now, your phone.",
    () =>
      "That's fine. Let's connect your phone next.",
    () =>
      "Okay then. Next up, connecting your phone.",
    () =>
      "No worries. Now let's link your phone.",
    () =>
      "Fair enough. Let's set up your phone.",
    () =>
      "Alright then. Time to connect your phone.",
    () =>
      "Okay. Let's get your phone ready.",
  ],
  go_google: [() => 'Opening Google. From here, use your keyboard or screen reader on the Google page.'],
  give_up: [() => 'I will stop listening now. You can use the buttons on the screen: press tab to move between them.'],
  connect_phone: [
    (code: string) =>
      `Now let us connect your phone. Open the camera on your phone and point it at the code on the screen, then tap the link that appears. When the page opens, tap the big button, and I will greet you. If the code cannot be scanned, open the address shown on the screen in the phone's browser, and type the code ${spelled(code)}.`,
    (code: string) =>
      `Next, your phone. Point your phone's camera at the code on the screen, and tap the link that pops up. When the page opens, tap the big Start button, and I'll say hello. Can't scan it? Open the address on the screen in your phone's browser, and type the code ${spelled(code)}.`,
    (code: string) =>
      `Let's link your phone. Open its camera, aim it at the code on the screen, and tap the link. Then tap the big Start button on the page that opens. If scanning doesn't work, type the address on the screen into your phone's browser, and enter the code ${spelled(code)}.`,
    (code: string) =>
      `Time to connect your phone. Open the phone's camera and hold it up to the code on the screen. Tap the link it shows, then tap the big Start button, and I'll greet you. If it won't scan, open the address on the screen in the phone's browser and type the code ${spelled(code)}.`,
    (code: string) =>
      `Now for your phone. Use its camera to scan the code on the screen, then tap the link. On the page that opens, tap the big Start button. No luck scanning? Type the address shown on the screen into the phone's browser, with the code ${spelled(code)}.`,
    (code: string) =>
      `Let's get your phone ready. Point the phone's camera at the code on the screen until a link appears, and tap it. Then tap the big Start button, and you'll hear me say hello. If that doesn't work, open the address on the screen in the phone's browser and type the code ${spelled(code)}.`,
    (code: string) =>
      `Your phone is next. Open the camera, point it at the code on the screen, and tap the link that comes up. Tap the big Start button on the new page, and I'll greet you. If the code won't scan, go to the address on the screen in the phone's browser and enter the code ${spelled(code)}.`,
    (code: string) =>
      `Okay, the phone. Scan the code on the screen with your phone's camera, then tap the link. When the page loads, tap the big Start button. If scanning is tricky, type the address on the screen into the phone's browser, and the code ${spelled(code)}.`,
    (code: string) =>
      `Now we'll connect your phone. Hold its camera up to the code on the screen and tap the link that appears. Then press the big Start button, and I'll say hello. Or, if it won't scan, open the address on the screen in your phone's browser and type the code ${spelled(code)}.`,
    (code: string) =>
      `Let's bring in your phone. Point its camera at the code on the screen and tap the link. On the page that opens, tap the big Start button, and I'll greet you. Can't scan? Type the address from the screen into the phone's browser, then the code ${spelled(code)}.`,
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
