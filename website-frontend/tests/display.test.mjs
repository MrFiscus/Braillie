// The display preferences (src/display/display.ts): colours and text size, and what happens with anything odd in storage.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const js = ts.transpileModule(readFileSync(new URL('../src/display/display.ts', import.meta.url), 'utf8'), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { resolveTheme, parsePrefs, DEFAULTS } = await import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'))

test('"match my device" becomes a real theme: more contrast wins, then dark, then light', () => {
  assert.equal(resolveTheme('auto', false, false), 'light')
  assert.equal(resolveTheme('auto', true, false), 'dark')
  assert.equal(resolveTheme('auto', true, true), 'contrast')
  assert.equal(resolveTheme('auto', false, true), 'contrast')
})

test('a chosen theme is never overruled by the device', () => {
  for (const s of ['light', 'dark', 'contrast']) assert.equal(resolveTheme(s, s !== 'dark', s !== 'contrast'), s)
})

test('saved preferences are read back, and nonsense falls back to the defaults', () => {
  assert.deepEqual(parsePrefs('{"scheme":"dark","size":"3"}'), { scheme: 'dark', size: '3' })
  assert.deepEqual(parsePrefs('{"scheme":"neon","size":"9"}'), DEFAULTS)
  assert.deepEqual(parsePrefs('{"scheme":"contrast"}'), { scheme: 'contrast', size: '1' })
  for (const bad of [null, '', 'not json', '[]', '5']) assert.deepEqual(parsePrefs(bad), DEFAULTS, String(bad))
})
