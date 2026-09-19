import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getMainStyles } from '../styles-react-components/mainStyles.tsx'
import AccSlider from '../styles-react-components/AccSlider.tsx'
import CustomButton from '../styles-react-components/CustomButton.tsx'

export interface StartPageProps {
  onStart?: () => void
}

export default function StartPage({ onStart }: StartPageProps) {
  const [scale] = useState<number>(1.0)
  const navigate = useNavigate()

  // Dynamic styles derived from mainStyles based on slider scale
  const styles = getMainStyles(scale)

  const handleStartClick = () => {
    if (onStart) {
      onStart()
    } else {
      navigate('/user-information')
    }
  }

  return (
    <main style={styles.container}>
      <div style={styles.card}>
        {/* Brand asset badge */}
        <div style={styles.brandBadge}>
          <span style={styles.brailleChar} aria-hidden="true">
            ⠃⠗⠁⠊⠇⠇⠊⠑
          </span>
          <span>Braillie</span>
        </div>

        {/* 1. First: The text "Read, Write, Learn" */}
        <h1 style={styles.title}>Read, Write, Learn</h1>

        {/* 2. Slider that makes every asset either larger or smaller */}
        <AccSlider />

        {/* 3. Button with text "Start Today" that redirects to GetToKnow */}
        <CustomButton
          buttonText="Start Today"
          onClick={handleStartClick}
        />
      </div>
    </main>
  )
}
