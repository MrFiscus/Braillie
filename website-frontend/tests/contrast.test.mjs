// The colours the site is built from, checked against the stylesheet itself: every pair that text (or a control) can appear in must meet
// WCAG AAA (7:1 for text) and 3:1 for shapes, in all three themes. Change a colour in braillie.css and this says whether it is still readable.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const css = readFileSync(new URL('../src/styles-css/braillie.css', import.meta.url), 'utf8')

const block = (selector) => {
  const start = css.indexOf(`${selector} {`)
  assert.ok(start >= 0, `no block for ${selector}`)
  const body = css.slice(css.indexOf('{', start) + 1, css.indexOf('}', start))
  return Object.fromEntries([...body.matchAll(/--([\w-]+):\s*([^;]+);/g)].map((m) => [m[1], m[2].trim()]))
}

const themes = {
  light: block(':root'),
  dark: { ...block(':root'), ...block(":root[data-theme='dark']") },
  'contrast-light': { ...block(':root'), ...block(":root[data-theme='contrast-light']") },
  contrast: { ...block(':root'), ...block(":root[data-theme='contrast']") },
}
const STRONG = new Set(['contrast', 'contrast-light']) // the two themes made for low vision

const rgb = (hex) => {
  assert.match(hex, /^#([0-9a-f]{3}|[0-9a-f]{6})$/i, `not a plain hex colour: ${hex}`)
  const full = hex.length === 4 ? '#' + [...hex.slice(1)].map((c) => c + c).join('') : hex
  return [1, 3, 5].map((i) => parseInt(full.slice(i, i + 2), 16))
}
const luminance = ([r, g, b]) => {
  const f = (v) => ((v /= 255) <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4)
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
}
const ratio = (a, b) => {
  const [hi, lo] = [luminance(rgb(a)), luminance(rgb(b))].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
}
const mix = (a, b, pct) => '#' + rgb(a).map((v, i) => Math.round((v * pct + rgb(b)[i] * (100 - pct)) / 100).toString(16).padStart(2, '0')).join('')

// [foreground, background, where it is used]
const TEXT = [
  ['ink', 'paper', 'body text'],
  ['ink', 'card', 'text on a sheet or card'],
  ['ink', 'paper-sunk', 'text in a note, a hovered key or the voice panel'],
  ['ink-soft', 'paper', 'secondary text'],
  ['ink-soft', 'card', 'secondary text on a card'],
  ['ink-soft', 'paper-sunk', 'secondary text in a note, a disabled key'],
  ['accent', 'paper', 'the eyebrow and the big letter'],
  ['accent', 'card', 'the eyebrow on a sheet'],
  ['accent', 'paper-sunk', 'emphasis inside a note'],
  ['on-accent', 'accent', 'the primary key'],
  ['paper', 'ink', 'a chosen option, the current letter, the skip link'],
  ['good', 'card', 'a good state on a card'],
  ['good', 'paper', 'a good state'],
  ['warn', 'paper-sunk', 'the emphasis in a problem note'],
  ['warn', 'paper', 'a problem'],
]

for (const [theme, t] of Object.entries(themes)) {
  test(`${theme}: text pairs meet AAA (7:1)`, () => {
    for (const [fg, bg, use] of TEXT) {
      const r = ratio(t[fg], t[bg])
      assert.ok(r >= 7, `${theme}: ${fg} ${t[fg]} on ${bg} ${t[bg]} is ${r.toFixed(2)}:1 (${use})`)
    }
  })

  // Beyond the AAA minimum: the reading text is comfortably above it (low vision needs margin, not just a pass), and in the two strong themes
  // every text pair is at least 9:1.
  test(`${theme}: reading text has real margin (ink 12:1, secondary text 10:1${STRONG.has(theme) ? ', every pair 9:1' : ''})`, () => {
    for (const bg of ['paper', 'card', 'paper-sunk']) {
      assert.ok(ratio(t.ink, t[bg]) >= 12, `${theme}: ink on ${bg} is ${ratio(t.ink, t[bg]).toFixed(2)}:1`)
      assert.ok(ratio(t['ink-soft'], t[bg]) >= 10, `${theme}: ink-soft on ${bg} is ${ratio(t['ink-soft'], t[bg]).toFixed(2)}:1`)
    }
    if (!STRONG.has(theme)) return
    for (const [fg, bg, use] of TEXT) assert.ok(ratio(t[fg], t[bg]) >= 9, `${theme}: ${fg} on ${bg} is ${ratio(t[fg], t[bg]).toFixed(2)}:1 (${use})`)
  })

  test(`${theme}: the hovered primary key stays AAA`, () => {
    const hover = mix(t.accent, t.ink, 86)
    assert.ok(ratio(t['on-accent'], hover) >= 7, `${theme}: on-accent on the hovered accent ${hover} is ${ratio(t['on-accent'], hover).toFixed(2)}:1`)
  })

  test(`${theme}: borders, dots and focus rings are clearly visible (3:1)`, () => {
    for (const [fg, bg] of [['ink', 'paper'], ['ink', 'card'], ['ink', 'paper-sunk'], ['accent', 'paper'], ['accent', 'card'], ['ink-soft', 'card'], ['good', 'card']]) {
      assert.ok(ratio(t[fg], t[bg]) >= 3, `${theme}: ${fg} on ${bg} is ${ratio(t[fg], t[bg]).toFixed(2)}:1`)
    }
  })
}

test('the QR code is always black on white, whatever the theme', () => {
  assert.match(css, /\.qr \{[^}]*background: #fff/)
})

test('the strong themes have thicker edges, and no theme has thinner ones than 3px', () => {
  assert.equal(block(':root')['bw'], '3px')
  for (const name of ['contrast', 'contrast-light']) assert.equal(block(`:root[data-theme='${name}']`)['bw'], '4px', name)
})

test('large text switches to one column, because media queries do not follow this page\'s text size', () => {
  assert.match(css, /:is\(html\[data-size='3'\], html\[data-size='4'\]\) \.page--split/)
  assert.match(css, /html\[data-size='4'\] \{\s*font-size: 190%/)
})

test('Windows forced-colours mode is handled', () => {
  assert.match(css, /@media \(forced-colors: active\)/)
})
