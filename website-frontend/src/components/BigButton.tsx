import type { ButtonHTMLAttributes, ReactNode } from 'react'
import '../styles-css/accessible.css'

interface BigButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary'
  children: ReactNode
}

/** A large, high-contrast button with a strong keyboard focus ring: for people who cannot see well or who use a keyboard or screen reader. */
const BigButton = ({ variant = 'primary', className = '', type = 'button', children, ...rest }: BigButtonProps) => (
  <button type={type} className={`a11y-btn ${variant === 'secondary' ? 'secondary' : ''} ${className}`.trim()} {...rest}>
    {children}
  </button>
)

export default BigButton
