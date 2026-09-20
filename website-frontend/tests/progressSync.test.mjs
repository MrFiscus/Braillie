// Tests for src/tutor/progressSync.ts with fake accounts and tutors. Run: node --test tests/
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const source = readFileSync(new URL('../src/tutor/progressSync.ts', import.meta.url), 'utf8')
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { ProgressSync, progressSignature, supabaseAccount } = await import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'))

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

function fakes({ user = 'u1', saved = null, loadError = null, saveError = null } = {}) {
  const log = { merged: [], saved: [], loads: 0 }
  const account = {
    userId: async () => user,
    load: async () => { log.loads++; if (loadError) throw new Error(loadError); return saved },
    save: async (id, data) => { if (saveError) throw new Error(saveError); log.saved.push([id, data]) },
  }
  const tutor = { get: async () => ({ letters: { a: 1 } }), merge: async (d) => { log.merged.push(d) } }
  return { account, tutor, log }
}

const summary = (over = {}) => ({ learned: 1, practised: 2, sessions: 1, streak: 1, best_streak: 1, mastery: { a: 0.25 }, confusions: [], lessons: {}, ...over })

test('signature is empty for nothing and changes whenever the progress does', () => {
  assert.equal(progressSignature(null), '')
  assert.equal(progressSignature(undefined), '')
  const base = progressSignature(summary())
  assert.equal(progressSignature(summary()), base, 'the same progress gives the same signature')
  for (const change of [{ sessions: 2 }, { practised: 3 }, { learned: 2 }, { streak: 2 }, { mastery: { a: 0.5 } }, { mastery: { a: 0.25, b: 0.25 } },
                        { lessons: { l1: { best: 0.8, last: 0.8, times: 1 } } }, { confusions: [{ touched: 'f', wanted: 'd', count: 1 }] }]) {
    assert.notEqual(progressSignature(summary(change)), base, JSON.stringify(change))
  }
})

test('start: signed out means nothing is merged and the learner is told progress stays on this computer', async () => {
  const { account, tutor, log } = fakes({ user: null })
  const seen = []
  const sync = new ProgressSync(account, tutor, (s, d) => seen.push([s, d]), 10)
  assert.equal(await sync.start(), 'signed-out')
  assert.equal(log.loads, 0)
  assert.deepEqual(log.merged, [])
  assert.match(sync.detail, /this computer only/)
  sync.schedule()
  await sleep(40)
  assert.deepEqual(log.saved, [], 'nothing is saved to an account that is not there')
})

test('start: the saved copy from the account is merged into the tutor, once', async () => {
  const saved = { letters: { q: 3 }, sessions: 4 }
  const { account, tutor, log } = fakes({ saved })
  const sync = new ProgressSync(account, tutor, () => {}, 10)
  assert.equal(await sync.start(), 'saved')
  assert.deepEqual(log.merged, [saved])
  await sync.start()
  await sync.start()
  assert.equal(log.loads, 1, 'asking again does nothing: the account is read once per session')
  assert.equal(log.merged.length, 1)
})

test('start: a brand new learner (nothing saved yet) merges nothing and is fine', async () => {
  const { account, tutor, log } = fakes({ saved: null })
  const sync = new ProgressSync(account, tutor, () => {}, 10)
  assert.equal(await sync.start(), 'saved')
  assert.deepEqual(log.merged, [])
})

test('start: an account that cannot be reached is reported, never thrown, and progress is still local', async () => {
  const { account, tutor, log } = fakes({ loadError: 'network down' })
  const sync = new ProgressSync(account, tutor, () => {}, 10)
  assert.equal(await sync.start(), 'error')
  assert.match(sync.detail, /network down/)
  assert.match(sync.detail, /still kept on this computer/)
  assert.deepEqual(log.merged, [])
})

test('a burst of changes is one save, of the latest copy, for the right learner', async () => {
  const { account, tutor, log } = fakes()
  const sync = new ProgressSync(account, tutor, () => {}, 30)
  await sync.start()
  for (let i = 0; i < 6; i++) { sync.schedule(); await sleep(5) }
  assert.equal(log.saved.length, 0, 'not yet: it waits for the answers to settle')
  await sleep(80)
  assert.equal(log.saved.length, 1)
  assert.deepEqual(log.saved[0], ['u1', { letters: { a: 1 } }])
  assert.equal(sync.status, 'saved')
})

test('a change during a save causes one more save afterwards, never two at once', async () => {
  let inFlight = 0, maxInFlight = 0, saves = 0
  const account = {
    userId: async () => 'u1', load: async () => null,
    save: async () => { inFlight++; maxInFlight = Math.max(maxInFlight, inFlight); await sleep(40); inFlight--; saves++ },
  }
  const tutor = { get: async () => ({}), merge: async () => {} }
  const sync = new ProgressSync(account, tutor, () => {}, 5)
  await sync.start()
  const first = sync.push()
  await sleep(10)
  await sync.push() // asked while the first is still saving
  await first
  await sleep(120)
  assert.equal(maxInFlight, 1)
  assert.equal(saves, 2, 'the first, then exactly one more for the change made meanwhile')
})

test('a failed save is reported with a message and can be retried', async () => {
  const { account, tutor } = fakes({ saveError: 'row level security' })
  const sync = new ProgressSync(account, tutor, () => {}, 10)
  await sync.start()
  assert.equal(await sync.push(), 'error')
  assert.match(sync.detail, /row level security/)
  const ok = fakes()
  const sync2 = new ProgressSync(ok.account, ok.tutor, () => {}, 10)
  await sync2.start()
  assert.equal(await sync2.push(), 'saved')
})

test('dispose cancels a pending save', async () => {
  const { account, tutor, log } = fakes()
  const sync = new ProgressSync(account, tutor, () => {}, 20)
  await sync.start()
  sync.schedule()
  sync.dispose()
  await sleep(60)
  assert.deepEqual(log.saved, [])
})

test('the Supabase adapter reads the session, the row and writes an upsert of the right shape', async () => {
  const calls = []
  const client = (over = {}) => ({
    auth: { getSession: async () => ({ data: { session: over.noUser ? null : { user: { id: 'abc' } } } }) },
    from: (table) => ({
      select: (cols) => ({ eq: (col, val) => ({ maybeSingle: async () => { calls.push(['select', table, cols, col, val]); return over.row ?? { data: null, error: null } } }) }),
      upsert: async (row) => { calls.push(['upsert', table, row]); return over.upsert ?? { error: null } },
    }),
  })
  const acct = supabaseAccount(client())
  assert.equal(await acct.userId(), 'abc')
  assert.equal(await supabaseAccount(client({ noUser: true })).userId(), null)
  assert.equal(await acct.load('abc'), null)
  assert.deepEqual(calls[0], ['select', 'learning_progress', 'data', 'user_id', 'abc'])
  assert.deepEqual(await supabaseAccount(client({ row: { data: { data: { x: 1 } }, error: null } })).load('abc'), { x: 1 })
  await assert.rejects(supabaseAccount(client({ row: { data: null, error: { message: 'no such table' } } })).load('abc'), /no such table/)
  await acct.save('abc', { y: 2 })
  const up = calls.at(-1)
  assert.equal(up[0], 'upsert')
  assert.equal(up[1], 'learning_progress')
  assert.deepEqual(Object.keys(up[2]).sort(), ['data', 'updated_at', 'user_id'])
  assert.deepEqual([up[2].user_id, up[2].data], ['abc', { y: 2 }])
  assert.ok(!Number.isNaN(Date.parse(up[2].updated_at)))
  await assert.rejects(supabaseAccount(client({ upsert: { error: { message: 'denied' } } })).save('abc', {}), /denied/)
})
