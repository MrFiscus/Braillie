import { useEffect, useRef, useState } from 'react'
import { SCHEMES, SIZES, applyPrefs, loadPrefs, savePrefs, type Prefs } from '../display/display.ts'
import { shareTheme } from '../tutor/tutorApi.ts'

// the colours offered: the device's own choice (which still turns to high contrast when the device asks for more), light, or dark
const SHOWN = SCHEMES.filter((s) => ['auto', 'light', 'dark'].includes(s.value))

/** "Display": text size and colours, remembered in this browser (and told to the tutor, so the phone's page matches; `share` false on
 * the landing page, which has no tutor). A plain disclosure: keyboard and screen-reader friendly. */
const DisplayMenu = ({ share = true }: { share?: boolean }) => {
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs)
  const box = useRef<HTMLDetailsElement>(null)

  useEffect(() => {
    const apply = () => {
      applyPrefs(prefs)
      if (share) shareTheme(document.documentElement.dataset.theme) // the phone's page takes the same colours
    }
    apply()
    if (prefs.scheme !== 'auto' || typeof window.matchMedia !== 'function') return
    const queries = [window.matchMedia('(prefers-color-scheme: dark)'), window.matchMedia('(prefers-contrast: more)')]
    const again = apply // the device changed its mind (dark at night, for instance)
    queries.forEach((q) => q.addEventListener('change', again))
    return () => queries.forEach((q) => q.removeEventListener('change', again))
  }, [prefs, share])

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
            {SHOWN.map((s) => (
              <label key={s.value}>
                <input type="radio" name="colours" value={s.value} checked={prefs.scheme === s.value} onChange={() => change({ ...prefs, scheme: s.value })} />
                <span>{s.label}</span>
              </label>
            ))}
          </div>
        </fieldset>
      </div>
    </details>
  )
}

export default DisplayMenu
