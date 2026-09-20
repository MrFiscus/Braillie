/** Is Google sign-in set up on this computer? False when the site is running on the placeholder values used only to let it start. */
export function googleSignInAvailable(): boolean {
  const key: string = import.meta.env.VITE_SUPABASE_ANON_KEY ?? ''
  const url: string = import.meta.env.VITE_SUPABASE_URL ?? ''
  return !!url && !!key && !key.startsWith('placeholder')
}
