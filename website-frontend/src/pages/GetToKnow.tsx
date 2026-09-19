import { useState, type FormEvent } from 'react'
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
  const styles = getToKnowStyles

  const handleCheckboxToggle = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id],
    )
  }

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()

    // what youre gonna do is iterate through selectedIds. put the numbers they 
    // selected into supabase as is-for-fun == TRUE etc etc. Boolean values.
    // there will a whole sequence of web pages for this. If any 
    //column is false, the page will simply be skipped and the user will not see
    // it. 
    const selectedMessage =
      selectedIds.length > 0 ? selectedIds.join(', ') : 'None'

    // Alert at the top of the screen with Submitted! and selected checkbox IDs
    alert(`Submitted!\nSelected IDs: ${selectedMessage}`)

    if (onSubmit) {
      onSubmit(selectedIds)
    }
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