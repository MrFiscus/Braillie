import { useState, type FormEvent } from 'react'
import { getToKnowStyles } from '../styles-react-components/mainStyles.tsx'
import CustomButton from '../styles-react-components/CustomButton.tsx'
import { supabase } from '../auths/supabaseClients.ts'
import { useNavigate } from 'react-router-dom'

interface ButtonOption {
  id: string
  label: string
}

const BUTTON_OPTIONS: ButtonOption[] = [
  { id: '1', label: 'I am learning braille for fun.' },
  { id: '2', label: 'I want to learn braille to communicate with a fully or parially blind family member' },
  { id: '3', label: 'I am an assistant or caretaker of a fully or partially blind loved one.' },
  { id: '4', label: 'I am partially blind.' },
  { id: '5', label: 'I am fully blind.' },
]

export interface GetToKnowProps {
  onSubmit?: (selectedIds: string[]) => void
}

const GetToKnow = ({ onSubmit }: GetToKnowProps) => {
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const navigate = useNavigate();


  const styles = getToKnowStyles


  // Allows selecting only ONE button at a time (toggles off if clicked again)
  const handleOptionToggle = (id: string) => {
    setSelectedIds((prev) => (prev.includes(id) ? [] : [id]))
  }

  const handleSubmit = async (e: FormEvent) => {
      e.preventDefault();

      if (selectedIds.length === 0) {
        alert('Please select an option.');
        return;
      }

      const { data: { user } } = await supabase.auth.getUser();

      if (!user) {
        alert('No authenticated user found.');
        return;
      
    }

    const purposeValue = parseInt(selectedIds[0], 10);

    // UPSERT: Inserts a new row if 'id' doesn't exist, otherwise updates the existing row
    const { error } = await supabase
      .from('profiles')
      .upsert({
        id: user.id,            // Primary key to match
        email: user.email,      // Optional extra metadata
        purpose: purposeValue,
      });

    if (error) {
      console.error('Error saving profile:', error.message);
      alert('Failed to save selection.');
      return;
    }

    if (onSubmit) {
      onSubmit(selectedIds);
    }
    navigate("/personalization")
  };

  return (
    <main style={styles.container}>
      <div style={styles.card}>
        <h1 style={styles.title}>Get to Know</h1>
        <p style={styles.subtitle}>Select any options that apply:</p>

        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.checkboxList}>
            {BUTTON_OPTIONS.map((option) => {
              const isSelected = selectedIds.includes(option.id)

              return (
                <CustomButton
                  key={option.id}
                  buttonText={option.label}
                  isSelected={isSelected}
                  onClick={() => handleOptionToggle(option.id)}
                />
              )
            })}
          </div>

          <CustomButton 
            buttonText="Next" 
            onClick={() => handleSubmit({ preventDefault: () => undefined } as FormEvent)}
           />
        </form>
      </div>
    </main>
  )
}

export default GetToKnow