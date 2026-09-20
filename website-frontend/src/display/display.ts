// How the site looks for this person: colours and text size. Kept in this browser (localStorage) and applied as attributes on <html>, which
// styles-css/braillie.css reads. index.html applies the saved choice before the first paint so the page never flashes the wrong colours.
export type Scheme = 'auto' | 'light' | 'dark' | 'contrast'
export type Size = '1' | '2' | '3'
export interface Prefs {
  scheme: Scheme
  size: Size
}

export const SCHEMES: { value: Scheme; label: string }[] = [
  { value: 'auto', label: 'Match my device' },
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
  { value: 'contrast', label: 'High contrast' },
]
export const SIZES: { value: Size; label: string }[] = [
  { value: '1', label: 'Standard' },
  { value: '2', label: 'Large' },
  { value: '3', label: 'Largest' },
]

const KEY = 'braillie.display'
export const DEFAULTS: Prefs = { scheme: 'auto', size: '1' }

/** "Match my device" becomes a real theme: high contrast if the device asks for more contrast, dark if it prefers dark, else light. */
export function resolveTheme(scheme: Scheme, prefersDark: boolean, prefersMoreContrast: boolean): Exclude<Scheme, 'auto'> {
  if (scheme !== 'auto') return scheme
  if (prefersMoreContrast) return 'contrast'
  return prefersDark ? 'dark' : 'light'
}

/** What was saved, defended against anything odd in storage. */
export function parsePrefs(raw: string | null): Prefs {
  try {
    const data = JSON.parse(raw ?? '{}') as Partial<Prefs>
    return {
      scheme: SCHEMES.some((s) => s.value === data.scheme) ? (data.scheme as Scheme) : DEFAULTS.scheme,
      size: SIZES.some((s) => s.value === data.size) ? (data.size as Size) : DEFAULTS.size,
    }
  } catch {
    return { ...DEFAULTS }
  }
}

export function loadPrefs(): Prefs {
  try {
    return parsePrefs(window.localStorage.getItem(KEY))
  } catch {
    return { ...DEFAULTS }
  }
}

export function savePrefs(p: Prefs): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(p))
  } catch {
    // not saved: it still applies for this visit
  }
}

const media = (q: string) => (typeof window.matchMedia === 'function' ? window.matchMedia(q) : null)

export function applyPrefs(p: Prefs): void {
  const root = document.documentElement
  root.dataset.theme = resolveTheme(p.scheme, !!media('(prefers-color-scheme: dark)')?.matches, !!media('(prefers-contrast: more)')?.matches)
  root.dataset.size = p.size
}
