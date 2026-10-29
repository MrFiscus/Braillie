import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { setMode, sharePage } from '../tutor/tutorApi.ts'
import { useTutorState } from '../tutor/useTutorState.ts'
import { useTutorSession } from '../tutor/useTutorSession.ts'
import { useUser } from '../auth/useUser.ts'
import { controlNames, press } from './pageControls.ts'
import { useVoiceTab } from './useVoiceTab.ts'

// Which page is open, as Braillo is told about it (see sharePage).
const PAGE_NAMES: Record<string, string> = {
  '/': 'landing',
  '/login': 'login',
  '/connect-phone': 'connect-phone',
  '/modes': 'modes',
  '/practice': 'practice',
  '/account': 'account',
  '/privacy': 'privacy',
  '/terms': 'terms',
  '/accessibility': 'accessibility',
}
const SCROLL = 0.8 // of the window's height, for "scroll down" / "scroll up"

/**
 * The whole website, by voice, on every page (it sits around all of them and shows nothing). It tells the tutor which page is open and
 * the names of every button, link and option on it, as a screen reader would read them; the tutor listens, and when someone says one
 * ("start learning", "dark", "privacy policy") or asks to go somewhere ("go back", "scroll down", "my page", "sign out"), it asks for
 * it here, and it is done.
 */
const VoiceControl = () => {
  const { pathname } = useLocation()
  const navigate = useNavigate()
  const { user, loading, signOut } = useUser()
  const { state, reach } = useTutorState(500)
  const [names, setNames] = useState<string[]>([])
  const voiceTab = useVoiceTab() // (with the website open in two tabs, only the one in use is driven by voice)
  useTutorSession(voiceTab ? user : undefined, loading, state) // (who is here: told by the tab in use only)

  // what is on the page, kept up to date as it changes (a menu opening, a lesson moving on)
  useEffect(() => {
    let timer = 0
    const look = () => {
      window.clearTimeout(timer)
      timer = window.setTimeout(
        () =>
          setNames((old) => {
            const now = controlNames()
            return now.join('|') === old.join('|') ? old : now
          }),
        250,
      )
    }
    look()
    const watch = new MutationObserver(look)
    watch.observe(document.body, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ['aria-label', 'disabled', 'open', 'hidden'],
    })
    return () => {
      watch.disconnect()
      window.clearTimeout(timer)
    }
  }, [pathname])

  // arriving (the page opened, the address changed, or this tab came to the front) is told once, so a page with no voice of its own is
  // announced each time someone arrives on it, not only the first time
  const page = PAGE_NAMES[pathname]
  const arriving = useRef(true)
  useEffect(() => {
    arriving.current = true
  }, [pathname, voiceTab])
  useEffect(() => {
    if (!page || reach !== 'yes' || !voiceTab) return
    sharePage(page, names, arriving.current)
    arriving.current = false
  }, [page, names, reach, voiceTab])

  // what the tutor asks for, once each (n counts up)
  const nav = state?.nav ?? null
  const lastNav = useRef<number | null>(null)
  useEffect(() => {
    if (!nav) return
    if (lastNav.current === null || nav.n < lastNav.current) {
      lastNav.current = nav.n // (asked for before this page opened: not for it)
      return
    }
    if (nav.n === lastNav.current) return
    lastNav.current = nav.n
    if (!voiceTab) return
    const go = (to: string | number) => void (typeof to === 'number' ? navigate(to) : navigate(to))
    switch (nav.to) {
      case 'login':
        go('/login')
        break
      case 'account':
        go('/account')
        break
      case 'signout':
        void signOut().then(() => go('/'))
        break
      case 'click':
        if (nav.target) press(nav.target)
        break
      case 'back':
        if (pathname === '/practice')
          void setMode('menu').then(() => go('/modes')) // (the activity ends: back to choosing)
        else if ((window.history.state?.idx ?? 0) > 0) go(-1)
        else go(user ? '/modes' : '/')
        break
      case 'home':
        go(user ? '/modes' : '/')
        break
      case 'scroll-down':
        window.scrollBy({
          top: window.innerHeight * SCROLL,
          behavior: 'smooth',
        })
        break
      case 'scroll-up':
        window.scrollBy({
          top: -window.innerHeight * SCROLL,
          behavior: 'smooth',
        })
        break
      case 'top':
        window.scrollTo({ top: 0, behavior: 'smooth' })
        break
    }
  }, [nav, navigate, pathname, signOut, user, voiceTab])

  return null
}

export default VoiceControl
