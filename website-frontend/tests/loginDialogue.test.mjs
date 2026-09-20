// The spoken sign-in conversation (src/voice/loginDialogue.ts): what is understood, and how the conversation goes.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const load = async (rel) => {
  const js = ts.transpileModule(readFileSync(new URL(rel, import.meta.url), 'utf8'), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
  return import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'))
}
const { parseChoice, parseBack, parseYesNo, parseName, runLoginDialogue, PROMPT_TEXT } = await load('../src/voice/loginDialogue.ts')

test('Google or guest, however it is said', () => {
  for (const t of ['Google', 'google.', 'sign in with Google please', 'Gogle', 'goggle', 'sign in', 'log in', 'gmail']) assert.equal(parseChoice(t), 'google', t)
  for (const t of ['guest', 'Guest!', 'continue without an account', 'no account', 'skip', 'as a guest', 'sign in as a guest', "I don't have one"]) assert.equal(parseChoice(t), 'guest', t)
  for (const t of ['what', 'say that again', 'help']) assert.equal(parseChoice(t), 'repeat', t)
  for (const t of ['banana', '', 'the weather is nice']) assert.equal(parseChoice(t), 'none', t)
})

test('a name is picked out of what people actually say', () => {
  const cases = { Sam: 'Sam', sam: 'Sam', 'my name is sam': 'Sam', "I'm Sam.": 'Sam', "it's priya, please": 'Priya', 'hi my name is José': 'José', "call me O'Neil": "O'Neil",
    'Mary-Ann': 'Mary-Ann', 'this is Lee': 'Lee', 'hello I am Ana Maria': 'Ana' }
  for (const [said, name] of Object.entries(cases)) assert.equal(parseName(said), name, said)
  for (const t of ['skip', 'Skip.', 'no name', "I'd rather not", 'never mind']) assert.equal(parseName(t), 'skip', t)
})

test('answers and chatter are not names', () => {
  for (const t of ['', '   ', 'yes', 'no', 'okay', 'Google', 'guest', 'hello', '12345', 'x'.repeat(40), 'um']) assert.equal(parseName(t), null, JSON.stringify(t))
})

test('yes, no, and something else', () => {
  for (const t of ['yes', 'Yeah.', 'yep', 'correct', "that's right", 'okay']) assert.equal(parseYesNo(t), 'yes', t)
  for (const t of ['no', 'nope', 'wrong', 'not quite']) assert.equal(parseYesNo(t), 'no', t)
  for (const t of ['Sam', 'my name is Sam', '']) assert.equal(parseYesNo(t), 'other', t)
})

test('someone already known can carry on or switch', () => {
  for (const t of ['continue', 'yes', 'carry on', 'that is me', 'okay']) assert.equal(parseBack(t), 'continue', t)
  for (const t of ['switch', 'no', 'not me', 'someone else', 'sign out']) assert.equal(parseBack(t), 'switch', t)
  assert.equal(parseBack('pardon'), 'repeat')
  assert.equal(parseBack('banana'), 'none')
})

// A scripted person: `heard` is what the microphone will hear, in order (null is silence). Records what Braillie said and did.
function conversation(heard, returning = null, maxMisses) {
  const log = { said: [], did: [], status: [] }
  const queue = [...heard]
  const ac = new AbortController()
  const channel = {
    say: async (prompt, who) => void log.said.push(who === undefined ? prompt : `${prompt}(${who})`),
    listen: async () => (queue.length ? queue.shift() : (ac.abort(), null)), // a person who has run out of things to say ends the test
  }
  const actions = {
    google: () => void log.did.push('google'),
    guest: (n) => void log.did.push(`guest:${n}`),
    carryOn: () => void log.did.push('carry-on'),
    switchAccount: () => void log.did.push('switch'),
  }
  const run = runLoginDialogue({ channel, actions, returning, signal: ac.signal, maxMisses, onStatus: (s) => log.status.push(s.phase) })
  return { log, run }
}

test('"Google" opens Google', async () => {
  const { log, run } = conversation(['sign in with google'])
  assert.equal(await run, 'google')
  assert.deepEqual(log.said, ['welcome', 'go_google'])
  assert.deepEqual(log.did, ['google'])
})

test('"guest", then a name, then "yes" continues as that guest', async () => {
  const { log, run } = conversation(['guest', 'my name is sam', 'yes'])
  assert.equal(await run, 'guest')
  assert.deepEqual(log.said, ['welcome', 'ask_name', 'confirm_name(Sam)', 'go_guest(Sam)'])
  assert.deepEqual(log.did, ['guest:Sam'])
})

test('a wrong name can be corrected, by "no" or by just saying it again', async () => {
  const a = conversation(['guest', 'pam', 'no', 'sam', 'yes'])
  assert.equal(await a.run, 'guest')
  assert.deepEqual(a.log.said, ['welcome', 'ask_name', 'confirm_name(Pam)', 'again_name', 'confirm_name(Sam)', 'go_guest(Sam)'])
  const b = conversation(['guest', 'pam', 'sam', 'yeah'])
  assert.equal(await b.run, 'guest')
  assert.deepEqual(b.log.did, ['guest:Sam'])
})

test('"skip" continues as a guest with no name, and nothing is confirmed that was not heard', async () => {
  const { log, run } = conversation(['guest', 'skip'])
  assert.equal(await run, 'guest')
  assert.deepEqual(log.said, ['welcome', 'ask_name', 'go_guest_anon'])
  assert.deepEqual(log.did, ['guest:'])
})

test('not understood: asked again, then it stops asking and points to the buttons', async () => {
  const { log, run } = conversation(['banana', null, 'the weather'])
  assert.equal(await run, 'gave-up')
  assert.deepEqual(log.said, ['welcome', 'retry_choice', 'retry_choice', 'give_up'])
  assert.deepEqual(log.did, [], 'it must not guess')
})

test('a wrong turn does not use up the patience for the next question', async () => {
  const { log, run } = conversation(['banana', 'guest', 'um', 'sam', 'yes'])
  assert.equal(await run, 'guest')
  assert.deepEqual(log.said, ['welcome', 'retry_choice', 'ask_name', 'retry_name', 'confirm_name(Sam)', 'go_guest(Sam)'])
})

test('"repeat" says the welcome again', async () => {
  const { log, run } = conversation(['what', 'google'])
  await run
  assert.deepEqual(log.said, ['welcome', 'welcome', 'go_google'])
})

test('someone already known: continue, or switch and start over', async () => {
  const a = conversation(['continue'], 'Ana')
  assert.equal(await a.run, 'carried-on')
  assert.deepEqual(a.log.said, ['welcome_back(Ana)'])
  assert.deepEqual(a.log.did, ['carry-on'])
  const b = conversation(['switch', 'guest', 'skip'], 'Ana')
  assert.equal(await b.run, 'guest')
  assert.deepEqual(b.log.did, ['switch', 'guest:'])
  assert.deepEqual(b.log.said, ['welcome_back(Ana)', 'welcome', 'ask_name', 'go_guest_anon'])
})

test('stopping (the person pressed a button) ends it quietly', async () => {
  const ac = new AbortController()
  const said = []
  const run = runLoginDialogue({
    channel: { say: async (p) => void said.push(p), listen: async () => (ac.abort(), null) },
    actions: { google() { throw new Error('no') }, guest() { throw new Error('no') }, carryOn() {}, switchAccount() {} },
    returning: null, signal: ac.signal, onStatus: () => {},
  })
  assert.equal(await run, 'stopped')
  assert.deepEqual(said, ['welcome'])
})

test('the browser voice says the same lines the tutor does', () => {
  const py = readFileSync(new URL('../../braille_tutor/tutor.py', import.meta.url), 'utf8')
  const block = py.slice(py.indexOf('PROMPTS = {'), py.indexOf('DIALOGUE_SECONDS'))
  const server = [...block.matchAll(/^\s{4}"(\w+)":/gm)].map((m) => m[1]).sort()
  assert.deepEqual(Object.keys(PROMPT_TEXT).sort(), server, 'add a line in both places (tutor.py PROMPTS and PROMPT_TEXT)')
  assert.match(PROMPT_TEXT.confirm_name('Sam'), /Sam/)
  for (const key of Object.keys(PROMPT_TEXT)) {
    const words = PROMPT_TEXT[key]('Sam').split(/\s+/).slice(0, 4).join(' ')
    assert.ok(block.replace(/"\s*\n\s*"/g, '').includes(words.replace('Sam', '{name}')) || block.includes(words), `${key} starts differently in tutor.py: ${words}`)
  }
})
