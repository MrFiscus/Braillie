import '../styles-css/App.css';


import { Routes, Route, useNavigate } from 'react-router-dom';

import StartPage from "../pages/StartPage.tsx"
import GetToKnow from './GetToKnow.tsx'
import Layout from "../components/Layout.tsx"
import {Login} from "../components/GoogleLogin.tsx"
import ConnectPhone from './ConnectPhone.tsx'
import Practice from './Practice.tsx'

function App() {
  const navigate = useNavigate()
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<StartPage />} />

        <Route path="get-started" element={<StartPage />} />
        <Route path="user-information" element={<GetToKnow onSubmit={() => navigate('/connect-phone')} />} />
        <Route path="connect-phone" element={<ConnectPhone />} />
        <Route path="practice" element={<Practice />} />
        <Route path="login" element={<Login />} />
        
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