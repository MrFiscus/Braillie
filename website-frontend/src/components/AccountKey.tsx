import { Link } from 'react-router-dom'
import { useUser } from '../auth/useUser.ts'

/** The round key at the top of the public pages: your initial, to your page, once you are signed in (or a guest); a person, to sign in, if not. */
const AccountKey = ({ className }: { className?: string }) => {
  const { user } = useUser()
  const label = user ? `Your page, ${user.name}` : 'Sign in'
  return (
    <Link className={className ? `accountkey ${className}` : 'accountkey'} to={user ? '/account' : '/login'} aria-label={label} title={label}>
      {user ? (
        <span aria-hidden="true">{user.name.charAt(0).toUpperCase()}</span>
      ) : (
        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <circle cx="12" cy="8" r="4" />
          <path d="M4.5 21c.8-4 3.8-6.5 7.5-6.5s6.7 2.5 7.5 6.5" />
        </svg>
      )}
    </Link>
  )
}

export default AccountKey
