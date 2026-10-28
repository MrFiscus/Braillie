// When to show the code to scan the phone again (pure: tested in tests/phoneWatch.test.mjs; the hook is usePhoneWatch.ts).

export const PHONE_PAGES = ['/modes', '/practice', '/account'] // the pages that need the phone
export const RECONNECT_AFTER_MS = 8000

/** Should the code to scan be shown again? Only on a page that needs the phone, for a phone that was linked, gone this long. */
export function shouldReconnect(pathname: string, everLinked: boolean, connected: boolean | undefined, goneForMs: number): boolean {
  return PHONE_PAGES.includes(pathname) && everLinked && connected === false && goneForMs >= RECONNECT_AFTER_MS
}

/** Where to go back to after reconnecting: one of the pages that need the phone, else home. */
export function backTo(search: string): string {
  const back = new URLSearchParams(search).get('back') ?? ''
  return PHONE_PAGES.includes(back) ? back : '/modes'
}
