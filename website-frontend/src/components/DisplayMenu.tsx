import { useEffect, useRef, useState } from 'react'
import { SCHEMES, SIZES, applyPrefs, loadPrefs, savePrefs, type Prefs } from '../display/display.ts'

/** "Display": text size and colours (including high contrast), remembered in this browser. A plain disclosure: keyboard and screen-reader friendly. */
const DisplayMenu = () => {
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs)
  const box = useRef<HTMLDetailsElement>(null)

  useEffect(() => {
    applyPrefs(prefs)
    if (prefs.scheme !== 'auto' || typeof window.matchMedia !== 'function') return
    const queries = [window.matchMedia('(prefers-color-scheme: dark)'), window.matchMedia('(prefers-contrast: more)')]
    const again = () => applyPrefs(prefs) // the device changed its mind (dark at night, for instance)
    queries.forEach((q) => q.addEventListener('change', again))
    return () => queries.forEach((q) => q.removeEventListener('change', again))
  }, [prefs])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && box.current?.open) {
        box.current.open = false
        box.current.querySelector('summary')?.focus()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [])

  const change = (next: Prefs) => {
    setPrefs(next)
    savePrefs(next)
  }

  return (
    <details className="display" ref={box}>
      <summary>Display</summary>
      <div className="display__panel">
        <fieldset>
          <legend>Text size</legend>
          <div className="seg">
            {SIZES.map((s) => (
              <label key={s.value}>
                <input type="radio" name="text-size" value={s.value} checked={prefs.size === s.value} onChange={() => change({ ...prefs, size: s.value })} />
                <span>{s.label}</span>
              </label>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>Colours</legend>
          <div className="seg">
            {SCHEMES.map((s) => (
              <label key={s.value}>
                <input type="radio" name="colours" value={s.value} checked={prefs.scheme === s.value} onChange={() => change({ ...prefs, scheme: s.value })} />
                <span>{s.label}</span>
              </label>
            ))}
          </div>
          <p className="hint small">Black on white and Yellow on black are the strongest: thicker edges, heavier text.</p>
        </fieldset>
      </div>
    </details>
  )
}

export default DisplayMenu
