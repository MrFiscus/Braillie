// Tests for src/auth/appUser.ts. Run: node --test "tests/*.test.mjs"
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const source = readFileSync(new URL('../src/auth/appUser.ts', import.meta.url), 'utf8')
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { firstName, userFromAccount, newGuest, forgetOldGuest, sessionFor, OLD_GUEST_KEY } = await import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'))

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

test('a guest is only a first name and an id for this visit, and each guest is a new visit', () => {
  const g = newGuest('Gina Rossi')
  assert.equal(g.kind, 'guest')
  assert.equal(g.name, 'Gina')
  assert.match(g.id, /^[A-Za-z0-9_-]{1,80}$/)
  assert.notEqual(newGuest('Gina').id, g.id, 'the next guest is another visit')
  assert.equal(newGuest('   ').name, 'friend', 'no name given: a friendly default')
})

test('a guest an older version stored is cleared away, and broken storage is no problem', () => {
  const s = new Storage()
  s.setItem(OLD_GUEST_KEY, JSON.stringify({ name: 'Old', id: 'abc' }))
  forgetOldGuest(s)
  assert.equal(s.getItem(OLD_GUEST_KEY), null)
  forgetOldGuest(null)
  forgetOldGuest({ removeItem() { throw new Error('blocked') } })
})

test('what is sent to the tutor: a google learner brings their account id, a guest brings nothing to keep', () => {
  assert.deepEqual(sessionFor({ kind: 'google', name: 'Ana', id: 'u1' }).body, { kind: 'google', name: 'Ana', profile: 'u1' })
  assert.deepEqual(sessionFor({ kind: 'guest', name: 'Gina' }).body, { kind: 'guest', name: 'Gina' })
  assert.deepEqual(sessionFor({ kind: 'guest', name: 'Guest', id: 'v1' }).body, { kind: 'guest', name: 'Guest', profile: 'v1' }, 'a visit id, to tell guests apart')
  assert.notEqual(sessionFor({ kind: 'guest', name: 'Guest', id: 'v1' }).key, sessionFor({ kind: 'guest', name: 'Guest', id: 'v2' }).key)
  assert.deepEqual(sessionFor({ kind: 'google', name: 'Ana' }).body, { kind: 'guest', name: 'Ana' }, 'a google user with no account id is treated as a guest: nothing is saved under nobody')
  const a = sessionFor({ kind: 'google', name: 'Ana', id: 'u1' }).key
  assert.notEqual(a, sessionFor({ kind: 'google', name: 'Ana', id: 'u2' }).key)
  assert.notEqual(a, sessionFor({ kind: 'guest', name: 'Ana' }).key)
  assert.equal(a, sessionFor({ kind: 'google', name: 'Ana', id: 'u1' }).key)
})
