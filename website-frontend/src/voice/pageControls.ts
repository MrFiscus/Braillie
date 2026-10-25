// Every control on the page, by the name a screen reader would read for it, so any of them can be used by voice: the tutor is told the
// names (see VoiceControl), and when someone says one, the page presses it here.

const SELECTOR = 'a[href], button, summary, [role="button"], [role="link"], input[type="radio"], input[type="checkbox"]'
const MAX = 60 // names told to the tutor (a page has far fewer)

/** A control's name: its aria-label, else the text of its label (for a radio or checkbox), else its own text, without what is only there
 * for screen readers (an "(opens in a new tab)" or a longer description). */
export function controlName(el: Element): string {
  const label = el.getAttribute('aria-label')
  if (label) return tidy(label)
  const source = el instanceof HTMLInputElement ? (el.labels?.[0] ?? el.closest('label')) : el
  if (!source) return ''
  const copy = source.cloneNode(true) as Element
  copy.querySelectorAll('.sr-only, svg, input, [aria-hidden="true"]').forEach((n) => n.remove())
  return tidy(copy.textContent ?? '')
}

const tidy = (text: string) =>
  text
    .replace(/[^\p{L}\p{N}' ,.-]/gu, ' ') // (said, not read: arrows, quotes and other symbols go)
    .replace(/\s+/g, ' ')
    .trim()

/** Can it be used? Not disabled, and not hidden from everyone (a closed Display menu still counts: it is opened first, see press). */
function usable(el: Element): boolean {
  if ((el as HTMLButtonElement).disabled) return false
  if (el.closest('[hidden], [aria-hidden="true"], [inert], .skip')) return false // (the skip link is for keyboards: by voice, there is nothing to skip)
  const style = window.getComputedStyle(el)
  return style.visibility !== 'hidden' && (style.display !== 'none' || !!el.closest('details:not([open])'))
}

function controls(root: ParentNode = document): Map<string, Element> {
  const found = new Map<string, Element>()
  // the page's own content first, then the top bar and the foot: "help" names the first few (what this page is for)
  const all = [...root.querySelectorAll(SELECTOR)]
  const inMain = (el: Element) => !!el.closest('main')
  for (const el of [...all.filter(inMain), ...all.filter((el) => !inMain(el))]) {
    if (!usable(el)) continue
    const name = controlName(el)
    const key = name.toLowerCase()
    if (name && name.length <= 60 && !found.has(key)) found.set(key, el)
    if (found.size >= MAX) break
  }
  return found
}

/** The names of the controls on the page now, for the tutor. */
export const controlNames = (): string[] => [...controls().values()].map(controlName)

/** Press the control with this name (as the tutor heard it): opens a closed menu it is in, moves the focus there, and clicks it.
 * Returns false if there is no such control now. */
export function press(name: string): boolean {
  const el = controls().get(name.toLowerCase())
  if (!el) return false
  const menu = el.closest('details:not([open])')
  if (menu && el.tagName !== 'SUMMARY') (menu as HTMLDetailsElement).open = true
  if (el instanceof HTMLElement) {
    el.focus({ preventScroll: false })
    el.scrollIntoView({ block: 'center', behavior: 'smooth' })
    el.click()
  }
  return true
}
