import type { CSSProperties } from 'react'

/**
 * Design tokens and theme configuration for Braillie
 */
export const themeTokens = {
  colors: {
    bgDark: '#0f172a',
    bgGradient: 'radial-gradient(ellipse at top, #1e1b4b 0%, #0f172a 70%)',
    cardBg: 'rgba(30, 41, 59, 0.85)',
    cardBorder: 'rgba(148, 163, 184, 0.25)',
    textPrimary: '#ffffff',
    textSecondary: '#cbd5e1',
    textMuted: '#94a3b8',
    primary: '#9333ea',
    primaryHover: '#7e22ce',
    primaryShadow: 'rgba(147, 51, 234, 0.45)',
    accent: '#a855f7',
    badgeBg: 'rgba(168, 85, 247, 0.18)',
    badgeBorder: 'rgba(168, 85, 247, 0.45)',
    badgeText: '#d8b4fe',
  },
  fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
}

/**
 * Returns dynamic, scale-responsive CSS properties for all start page assets.
 * As the user adjusts the scale slider, every visual element scales proportionally.
 *
 * @param scale Multiplier where 1.0 is 100% normal size
 */
export function getMainStyles(scale: number = 1.0) {
  const clampedScale = Math.max(0.6, Math.min(2.5, scale))

  const container: CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '100vh',
    width: '100%',
    padding: `${Math.round(24 * clampedScale)}px`,
    boxSizing: 'border-box',
    fontFamily: themeTokens.fontFamily,
    background: themeTokens.colors.bgGradient,
    backgroundColor: themeTokens.colors.bgDark,
    color: themeTokens.colors.textPrimary,
    textAlign: 'center',
    overflowX: 'hidden',
  }

  const card: CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    width: '100%',
    maxWidth: `${Math.round(520 * clampedScale)}px`,
    padding: `${Math.round(44 * clampedScale)}px ${Math.round(40 * clampedScale)}px`,
    backgroundColor: themeTokens.colors.cardBg,
    backdropFilter: 'blur(16px)',
    WebkitBackdropFilter: 'blur(16px)',
    border: `${Math.max(1, Math.round(1.5 * clampedScale))}px solid ${themeTokens.colors.cardBorder}`,
    borderRadius: `${Math.round(24 * clampedScale)}px`,
    boxShadow: `0 ${Math.round(20 * clampedScale)}px ${Math.round(40 * clampedScale)}px rgba(0, 0, 0, 0.45)`,
    gap: `${Math.round(24 * clampedScale)}px`,
    boxSizing: 'border-box',
    transition: 'all 0.12s ease-out',
  }

  const brandBadge: CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: `${Math.round(10 * clampedScale)}px`,
    padding: `${Math.round(6 * clampedScale)}px ${Math.round(18 * clampedScale)}px`,
    borderRadius: `${Math.round(999 * clampedScale)}px`,
    backgroundColor: themeTokens.colors.badgeBg,
    border: `${Math.max(1, Math.round(1 * clampedScale))}px solid ${themeTokens.colors.badgeBorder}`,
    color: themeTokens.colors.badgeText,
    fontSize: `${Math.round(15 * clampedScale)}px`,
    fontWeight: 600,
    letterSpacing: '0.04em',
  }

  const brailleChar: CSSProperties = {
    fontSize: `${Math.round(18 * clampedScale)}px`,
    lineHeight: 1,
    letterSpacing: `${Math.round(2 * clampedScale)}px`,
  }

  const title: CSSProperties = {
    margin: 0,
    fontSize: `${Math.round(36 * clampedScale)}px`,
    fontWeight: 800,
    letterSpacing: '-0.025em',
    lineHeight: 1.25,
    color: themeTokens.colors.textPrimary,
    textShadow: '0 2px 10px rgba(0, 0, 0, 0.35)',
  }

  const sliderSection: CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    width: '100%',
    gap: `${Math.round(12 * clampedScale)}px`,
    margin: `${Math.round(8 * clampedScale)}px 0`,
    padding: `${Math.round(18 * clampedScale)}px`,
    backgroundColor: 'rgba(15, 23, 42, 0.6)',
    borderRadius: `${Math.round(16 * clampedScale)}px`,
    border: `${Math.max(1, Math.round(1 * clampedScale))}px solid rgba(148, 163, 184, 0.15)`,
    boxSizing: 'border-box',
  }

  const sliderHeader: CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%',
  }

  const sliderLabel: CSSProperties = {
    fontSize: `${Math.round(16 * clampedScale)}px`,
    fontWeight: 600,
    color: themeTokens.colors.textSecondary,
    cursor: 'pointer',
  }

  const sliderValueBadge: CSSProperties = {
    fontSize: `${Math.round(14 * clampedScale)}px`,
    fontWeight: 700,
    color: themeTokens.colors.badgeText,
    backgroundColor: themeTokens.colors.badgeBg,
    padding: `${Math.round(3 * clampedScale)}px ${Math.round(10 * clampedScale)}px`,
    borderRadius: `${Math.round(8 * clampedScale)}px`,
    border: `${Math.max(1, Math.round(1 * clampedScale))}px solid ${themeTokens.colors.badgeBorder}`,
  }

  const sliderInput: CSSProperties = {
    width: '100%',
    height: `${Math.round(10 * clampedScale)}px`,
    accentColor: themeTokens.colors.accent,
    cursor: 'pointer',
    margin: `${Math.round(4 * clampedScale)}px 0`,
  }

  const sliderIndicators: CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    width: '100%',
    fontSize: `${Math.round(12 * clampedScale)}px`,
    color: themeTokens.colors.textMuted,
    userSelect: 'none',
  }

  const button: CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    padding: `${Math.round(16 * clampedScale)}px ${Math.round(28 * clampedScale)}px`,
    fontSize: `${Math.round(18 * clampedScale)}px`,
    fontWeight: 700,
    letterSpacing: '0.02em',
    color: themeTokens.colors.textPrimary,
    backgroundColor: themeTokens.colors.primary,
    border: 'none',
    borderRadius: `${Math.round(14 * clampedScale)}px`,
    cursor: 'pointer',
    boxShadow: `0 ${Math.round(8 * clampedScale)}px ${Math.round(20 * clampedScale)}px ${themeTokens.colors.primaryShadow}`,
    transition: 'all 0.15s ease',
    boxSizing: 'border-box',
    textDecoration: 'none',
  }

  const buttonHover: CSSProperties = {
    backgroundColor: themeTokens.colors.primaryHover,
    transform: 'translateY(-2px)',
    boxShadow: `0 ${Math.round(12 * clampedScale)}px ${Math.round(26 * clampedScale)}px ${themeTokens.colors.primaryShadow}`,
  }

  return {
    container,
    card,
    brandBadge,
    brailleChar,
    title,
    sliderSection,
    sliderHeader,
    sliderLabel,
    sliderValueBadge,
    sliderInput,
    sliderIndicators,
    button,
    buttonHover,
  }
}

/**
 * Returns scale-responsive styles for the GetToKnow checkbox page.
 */
export function getGetToKnowStyles(scale: number = 1.0) {
  const clampedScale = Math.max(0.6, Math.min(2.5, scale))

  const container: CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '100vh',
    width: '100%',
    padding: `${Math.round(24 * clampedScale)}px`,
    boxSizing: 'border-box',
    fontFamily: themeTokens.fontFamily,
    background: themeTokens.colors.bgGradient,
    backgroundColor: themeTokens.colors.bgDark,
    color: themeTokens.colors.textPrimary,
    textAlign: 'center',
    overflowX: 'hidden',
  }

  const card: CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    width: '100%',
    maxWidth: `${Math.round(520 * clampedScale)}px`,
    padding: `${Math.round(44 * clampedScale)}px ${Math.round(40 * clampedScale)}px`,
    backgroundColor: themeTokens.colors.cardBg,
    backdropFilter: 'blur(16px)',
    WebkitBackdropFilter: 'blur(16px)',
    border: `${Math.max(1, Math.round(1.5 * clampedScale))}px solid ${themeTokens.colors.cardBorder}`,
    borderRadius: `${Math.round(24 * clampedScale)}px`,
    boxShadow: `0 ${Math.round(20 * clampedScale)}px ${Math.round(40 * clampedScale)}px rgba(0, 0, 0, 0.45)`,
    gap: `${Math.round(20 * clampedScale)}px`,
    boxSizing: 'border-box',
  }

  const title: CSSProperties = {
    margin: 0,
    fontSize: `${Math.round(32 * clampedScale)}px`,
    fontWeight: 800,
    letterSpacing: '-0.025em',
    color: themeTokens.colors.textPrimary,
  }

  const subtitle: CSSProperties = {
    margin: 0,
    fontSize: `${Math.round(15 * clampedScale)}px`,
    color: themeTokens.colors.textSecondary,
    fontWeight: 400,
  }

  const form: CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    width: '100%',
    gap: `${Math.round(14 * clampedScale)}px`,
  }

  const checkboxList: CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    width: '100%',
    gap: `${Math.round(10 * clampedScale)}px`,
    textAlign: 'left',
  }

  const checkboxItem: CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: `${Math.round(14 * clampedScale)}px`,
    padding: `${Math.round(14 * clampedScale)}px ${Math.round(18 * clampedScale)}px`,
    borderRadius: `${Math.round(12 * clampedScale)}px`,
    backgroundColor: 'rgba(15, 23, 42, 0.65)',
    border: `${Math.max(1, Math.round(1.5 * clampedScale))}px solid rgba(148, 163, 184, 0.2)`,
    cursor: 'pointer',
    transition: 'all 0.15s ease',
    userSelect: 'none',
  }

  const checkboxItemActive: CSSProperties = {
    backgroundColor: 'rgba(168, 85, 247, 0.18)',
    borderColor: themeTokens.colors.accent,
    boxShadow: `0 0 ${Math.round(12 * clampedScale)}px rgba(168, 85, 247, 0.25)`,
  }

  const checkboxInput: CSSProperties = {
    width: `${Math.round(20 * clampedScale)}px`,
    height: `${Math.round(20 * clampedScale)}px`,
    accentColor: themeTokens.colors.accent,
    cursor: 'pointer',
  }

  const checkboxLabel: CSSProperties = {
    fontSize: `${Math.round(16 * clampedScale)}px`,
    fontWeight: 600,
    color: themeTokens.colors.textPrimary,
    cursor: 'pointer',
    flex: 1,
  }

  const submitButton: CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    padding: `${Math.round(16 * clampedScale)}px ${Math.round(28 * clampedScale)}px`,
    fontSize: `${Math.round(18 * clampedScale)}px`,
    fontWeight: 700,
    letterSpacing: '0.02em',
    color: themeTokens.colors.textPrimary,
    backgroundColor: themeTokens.colors.primary,
    border: 'none',
    borderRadius: `${Math.round(14 * clampedScale)}px`,
    cursor: 'pointer',
    boxShadow: `0 ${Math.round(8 * clampedScale)}px ${Math.round(20 * clampedScale)}px ${themeTokens.colors.primaryShadow}`,
    transition: 'all 0.15s ease',
    marginTop: `${Math.round(8 * clampedScale)}px`,
    boxSizing: 'border-box',
  }

  const submitButtonHover: CSSProperties = {
    backgroundColor: themeTokens.colors.primaryHover,
    transform: 'translateY(-2px)',
    boxShadow: `0 ${Math.round(12 * clampedScale)}px ${Math.round(26 * clampedScale)}px ${themeTokens.colors.primaryShadow}`,
  }

  return {
    container,
    card,
    title,
    subtitle,
    form,
    checkboxList,
    checkboxItem,
    checkboxItemActive,
    checkboxInput,
    checkboxLabel,
    submitButton,
    submitButtonHover,
  }
}

/**
 * Base styles (scale 1.0)
 */
export const mainStyles = {
  ...getMainStyles(1.0),
  getToKnow: getGetToKnowStyles(1.0),
}
export const getToKnowStyles = getGetToKnowStyles(1.0)
export default mainStyles
