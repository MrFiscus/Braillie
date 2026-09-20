import '../styles-css/App.css';


import { Routes, Route, Navigate } from 'react-router-dom';

import Layout from "../components/Layout.tsx"
import SignIn from './SignIn.tsx'
import Modes from './Modes.tsx'
import ConnectPhone from './ConnectPhone.tsx'
import Practice from './Practice.tsx'

// The one way through the app: sign in -> link the phone -> choose Learn, Read or Quiz -> do it. (The earlier start page and Get to Know
// pages are no longer part of it; their files are still in this folder.)
function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/login" replace />} />

        <Route path="connect-phone" element={<ConnectPhone />} />
        <Route path="practice" element={<Practice />} />
        <Route path="login" element={<SignIn />} />
        <Route path="modes" element={<Modes />} />
        <Route path="*" element={<Navigate to="/login" replace />} /> {/* any old or unknown address starts again at the sign-in page */}
        
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
  );
}

export default App;