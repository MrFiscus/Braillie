// When the phone drops out for good, the code to scan it again comes back (src/tutor/phoneWatch.ts).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const load = async (rel) => {
  const js = ts.transpileModule(readFileSync(new URL(rel, import.meta.url), 'utf8'), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
  return import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'))
}
const { shouldReconnect, backTo, RECONNECT_AFTER_MS } = await load('../src/tutor/phoneWatch.ts')

test('the code comes back only for a linked phone gone a while, on a page that needs it', () => {
  assert.equal(shouldReconnect('/practice', true, false, RECONNECT_AFTER_MS), true)
  assert.equal(shouldReconnect('/practice', true, false, 2000), false, 'a wobble (a camera restarting) is not the end')
  assert.equal(shouldReconnect('/practice', false, false, RECONNECT_AFTER_MS), false, 'never linked yet: that is the connect page itself')
  assert.equal(shouldReconnect('/practice', true, true, RECONNECT_AFTER_MS), false)
  assert.equal(shouldReconnect('/practice', true, undefined, RECONNECT_AFTER_MS), false, 'no phone at all (the laptop camera)')
  assert.equal(shouldReconnect('/privacy', true, false, RECONNECT_AFTER_MS), false, 'reading the privacy policy needs no phone')
})

test('after reconnecting, back to where they were (a page that needs the phone), else home', () => {
  assert.equal(backTo('?back=%2Fpractice'), '/practice')
  assert.equal(backTo('?back=/account'), '/account')
  assert.equal(backTo('?back=https://evil.example'), '/modes')
  assert.equal(backTo(''), '/modes')
})
