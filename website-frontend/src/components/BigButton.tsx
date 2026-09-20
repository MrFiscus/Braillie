import type { ButtonHTMLAttributes, ReactNode } from 'react'

interface BigButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary'
  quiet?: boolean // smaller, for a side action
  children: ReactNode
}

/** A big key: at least 72 px tall, a thick border, a strong focus ring (see .btn in braillie.css). Nothing depends on hover or colour. */
const BigButton = ({ variant = 'primary', quiet = false, className = '', type = 'button', children, ...rest }: BigButtonProps) => (
  <button type={type} className={['btn', variant === 'primary' ? 'btn--primary' : '', quiet ? 'btn--quiet' : '', className].filter(Boolean).join(' ')} {...rest}>
    {children}
  </button>
)

export default BigButton
