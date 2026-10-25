// What people say on their page (src/voice/accountVoice.ts).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const load = async (rel) => {
  const js = ts.transpileModule(readFileSync(new URL(rel, import.meta.url), 'utf8'), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
  return import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'))
}
const { parseAccount } = await load('../src/voice/accountVoice.ts')

test('what people actually say on their page is understood', () => {
  const cases = {
    continue: 'home', 'carry on please': 'home', 'go home': 'home', 'okay': 'home',
    letters: 'letters', 'which letters do I know': 'letters', 'read my letters': 'letters', 'how am I doing': 'letters', 'my alphabet': 'letters',
    'sign out': 'signout', 'log me out': 'signout', 'Sign-out.': 'signout',
    learn: 'learn', "let's do a lesson": 'learn', read: 'read', 'quiz me': 'quiz',
    repeat: 'repeat', 'say that again': 'repeat', help: 'help', 'what can I say': 'help',
    banana: 'none', '': 'none', '   ': 'none',
  }
  for (const [said, want] of Object.entries(cases)) assert.equal(parseAccount(said), want, said)
})

test('the tutor has a line for everything the page asks it to say', () => {
  const py = readFileSync(new URL('../../braille_tutor/tutor.py', import.meta.url), 'utf8')
  const page = readFileSync(new URL('../src/voice/useAccountVoice.ts', import.meta.url), 'utf8') + readFileSync(new URL('../src/pages/Account.tsx', import.meta.url), 'utf8')
  for (const [, name] of page.matchAll(/sayPrompt\('(\w+)'/g)) assert.match(py, new RegExp(`^\\s+"${name}": `, 'm'), name)
})
