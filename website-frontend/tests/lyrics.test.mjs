// Following a spoken line sentence by sentence (src/voice/lyrics.ts): how the time is shared out, and where speech is at a moment.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const js = ts.transpileModule(readFileSync(new URL('../src/voice/lyrics.ts', import.meta.url), 'utf8'), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { timeline, position } = await import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'))

const WELCOME = 'Welcome to Braillie. On this page you can sign in with Google, or continue without an account. To sign in, say Google.'

test('a line is split into its sentences, which share the whole time in order', () => {
  const line = timeline(WELCOME)
  assert.deepEqual(line.map((s) => s.words.join(' ')), ['Welcome to Braillie.', 'On this page you can sign in with Google, or continue without an account.', 'To sign in, say Google.'])
  assert.equal(line[0].start, 0)
  assert.equal(line.at(-1).end, 1)
  for (let i = 1; i < line.length; i++) assert.equal(line[i].start, line[i - 1].end)
  assert.ok(line[1].end - line[1].start > line[0].end - line[0].start, 'a longer sentence takes longer')
})

test('as time passes, the sentence moves on and its words fill in', () => {
  const line = timeline(WELCOME)
  assert.deepEqual(position(line, 0), { sentence: 0, said: 0 })
  const mid = position(line, (line[1].start + line[1].end) / 2)
  assert.equal(mid.sentence, 1)
  assert.ok(mid.said > 0 && mid.said < line[1].words.length)
  assert.deepEqual(position(line, 1), { sentence: 2, said: line[2].words.length })
  assert.deepEqual(position(line, 7), { sentence: 2, said: line[2].words.length }, 'past the end: the last sentence, all said')
  assert.deepEqual(position(line, -1), { sentence: 0, said: 0 })
})

test('a line without full stops is one sentence, and an empty line is nothing', () => {
  assert.equal(timeline('Okay').length, 1)
  assert.deepEqual(position(timeline(''), 0.5), { sentence: 0, said: 0 })
})
