// Tests for src/auth/appUser.ts. Run: node --test "tests/*.test.mjs"
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const source = readFileSync(new URL('../src/auth/appUser.ts', import.meta.url), 'utf8')
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { firstName, userFromAccount, loadGuest, saveGuest, clearGuest, sessionFor, GUEST_KEY } = await import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'))

class Storage {
  constructor() { this.m = new Map() }
  getItem(k) { return this.m.has(k) ? this.m.get(k) : null }
  setItem(k, v) { this.m.set(k, v) }
  removeItem(k) { this.m.delete(k) }
}

test('the first name is what the tutor calls someone, tidied', () => {
  assert.equal(firstName('Smaran Pokharel'), 'Smaran')
  assert.equal(firstName('  ana   maria '), 'ana')
  assert.equal(firstName("Zoë O'Neil"), 'Zoë')
  assert.equal(firstName('<b>Bob</b>'), 'bBobb', 'markup characters are dropped')
  assert.equal(firstName('José'), 'José', 'accented letters are kept')
  assert.equal(firstName('李小龍'), '李小龍', 'so are other scripts')
  assert.equal(firstName(''), '')
  assert.equal(firstName(null), '')
  assert.equal(firstName(undefined), '')
  assert.equal(firstName('x'.repeat(100)).length, 40)
})

test('a Google account: given name first, then the full name, then the email, then a friendly default', () => {
  assert.deepEqual(userFromAccount({ id: 'u1', user_metadata: { given_name: 'Ana', full_name: 'Ana Maria Lopez' } }), { kind: 'google', name: 'Ana', id: 'u1' })
  assert.deepEqual(userFromAccount({ id: 'u2', user_metadata: { full_name: 'Ben Carter' } }), { kind: 'google', name: 'Ben', id: 'u2' })
  assert.deepEqual(userFromAccount({ id: 'u3', user_metadata: { name: 'Cara D' } }), { kind: 'google', name: 'Cara', id: 'u3' })
  assert.deepEqual(userFromAccount({ id: 'u4', email: 'dan.smith@example.com', user_metadata: {} }), { kind: 'google', name: 'dan.smith', id: 'u4' })
  assert.equal(userFromAccount({ id: 'u5' }).name, 'friend')
  assert.equal(userFromAccount({ id: 'u6', user_metadata: { given_name: 42 } }).name, 'friend', 'a non-text name is ignored, not crashed on')
})

test('a guest is remembered for this tab and only ever by first name', () => {
  const s = new Storage()
  assert.equal(loadGuest(s), null)
  const g = saveGuest(s, 'Gina Rossi')
  assert.deepEqual(g, { kind: 'guest', name: 'Gina' })
  assert.deepEqual(JSON.parse(s.getItem(GUEST_KEY)), { name: 'Gina' }, 'nothing but the name is stored')
  assert.deepEqual(loadGuest(s), { kind: 'guest', name: 'Gina' })
  assert.equal(saveGuest(s, '   ').name, 'friend', 'no name given: a friendly default')
  clearGuest(s)
  assert.equal(loadGuest(s), null)
})

test('storage that is missing, broken or full never stops a guest continuing', () => {
  assert.equal(loadGuest(null), null)
  assert.deepEqual(saveGuest(null, 'Ana'), { kind: 'guest', name: 'Ana' })
  clearGuest(null)
  const broken = { getItem() { throw new Error('blocked') }, setItem() { throw new Error('full') }, removeItem() { throw new Error('blocked') } }
  assert.equal(loadGuest(broken), null)
  assert.deepEqual(saveGuest(broken, 'Ana'), { kind: 'guest', name: 'Ana' })
  clearGuest(broken)
  for (const junk of ['not json', '{"name": 5}', '{}', 'null', '"x"']) {
    const s = new Storage()
    s.setItem(GUEST_KEY, junk)
    assert.equal(loadGuest(s), null, junk)
  }
})

test('what is sent to the tutor: a google learner brings their account id, a guest brings nothing to keep', () => {
  assert.deepEqual(sessionFor({ kind: 'google', name: 'Ana', id: 'u1' }).body, { kind: 'google', name: 'Ana', profile: 'u1' })
  assert.deepEqual(sessionFor({ kind: 'guest', name: 'Gina' }).body, { kind: 'guest', name: 'Gina' })
  assert.deepEqual(sessionFor({ kind: 'google', name: 'Ana' }).body, { kind: 'guest', name: 'Ana' }, 'a google user with no account id is treated as a guest: nothing is saved under nobody')
  const a = sessionFor({ kind: 'google', name: 'Ana', id: 'u1' }).key
  assert.notEqual(a, sessionFor({ kind: 'google', name: 'Ana', id: 'u2' }).key)
  assert.notEqual(a, sessionFor({ kind: 'guest', name: 'Ana' }).key)
  assert.equal(a, sessionFor({ kind: 'google', name: 'Ana', id: 'u1' }).key)
})
