import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { getToKnowStyles } from '../styles-react-components/mainStyles.tsx'
import CustomButton from '../styles-react-components/CustomButton.tsx'

interface CheckboxOption {
  id: string
  label: string
}

const CHECKBOX_OPTIONS: CheckboxOption[] = [
  { id: 'for-fun', label: 'I am learning braille for fun.' },
  { id: 'communicate', label: 'I want to learn braille to communicate with a fully or parially blind family member' },
  { id: 'caretaker', label: 'I am an assistant or caretaker of a fully or partially blind loved one.' },
  { id: 'partial', label: 'I am partially blind.' },
  { id: 'full', label: 'I am fully blind.' },
]

export interface GetToKnowProps {
  onSubmit?: (selectedIds: string[]) => void
}

const GetToKnow = ({ onSubmit }: GetToKnowProps) => {
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const navigate = useNavigate()
  const styles = getToKnowStyles

  const handleCheckboxToggle = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id],
    )
  }

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()

    // TODO: persist selectedIds to Supabase before navigating
    // (iterate selectedIds, write boolean columns is-for-fun, communicate, etc.)
    if (onSubmit) {
      onSubmit(selectedIds)
    }

    navigate('/login')
  }

  return (
    <main style={styles.container}>
      <div style={styles.card}>
        <h1 style={styles.title}>Get to Know</h1>
        <p style={styles.subtitle}>Select any options that apply:</p>

        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.checkboxList}>
            {CHECKBOX_OPTIONS.map((option) => {
              const isChecked = selectedIds.includes(option.id)

              return (
                <label
                  key={option.id}
                  htmlFor={option.id}
                  style={{
                    ...styles.checkboxItem,
                    ...(isChecked ? styles.checkboxItemActive : {}),
                  }}
                >
                  <input
                    id={option.id}
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => handleCheckboxToggle(option.id)}
                    style={styles.checkboxInput}
                  />
                  <span style={styles.checkboxLabel}>{option.label}</span>
                </label>
              )
            })}
          </div>

          <CustomButton
            buttonText='Next'
            onClick={() => handleSubmit({ preventDefault: () => undefined } as FormEvent)}
          />
        </form>
      </div>
    </main>
  )
}

export default GetToKnow