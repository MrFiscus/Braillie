import '../styles-css/braillie.css';


import { Routes, Route, Navigate } from 'react-router-dom';

import Shell from '../components/Shell.tsx'
import SignIn from './SignIn.tsx'
import Modes from './Modes.tsx'
import ConnectPhone from './ConnectPhone.tsx'
import Practice from './Practice.tsx'
import Landing from './Landing.tsx'
import Account from './Account.tsx'
import { Accessibility, Privacy, Terms } from './Legal.tsx'
import VoiceControl from '../voice/VoiceControl.tsx'

// Everything can be done by voice too, on every page (VoiceControl: it shows nothing).
// The way through the app: the landing page (what Braillie is) -> sign in -> link the phone -> home: Learn, Read or Quiz -> do it.
function App() {
  return (
    <>
    <VoiceControl />
    <Routes>
      <Route index element={<Landing />} />
      <Route path="privacy" element={<Privacy />} />
      <Route path="terms" element={<Terms />} />
      <Route path="accessibility" element={<Accessibility />} />
      <Route element={<Shell />}>

        <Route path="connect-phone" element={<ConnectPhone />} />
        <Route path="practice" element={<Practice />} />
        <Route path="login" element={<SignIn />} />
        <Route path="modes" element={<Modes />} />
        <Route path="account" element={<Account />} />
        <Route path="*" element={<Navigate to="/" replace />} /> {/* any old or unknown address starts again at the front door */}
        
        {/*gaslight gatekeep girlboss */}
        {/* here purely for reference, from another project I did
        <Route 
          path="editor-dashboard" 
          element={
            <RequireRole allowedRoles={['editor', 'admin']}>
              <EditorDashboard />
            </RequireRole>
            }
          />*/}
      </Route>
      
    </Routes>
    </>
  );
}

export default App;