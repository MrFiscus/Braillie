import '../styles-css/App.css';


import { Routes, Route } from 'react-router-dom';

import StartPage from "../pages/StartPage.tsx"
import GetToKnow from './GetToKnow.tsx'
import Layout from "../components/Layout.tsx"
import {Login} from "../components/GoogleLogin.tsx"

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<StartPage />} />

        <Route path="get-started" element={<StartPage />} />
        <Route path="user-information" element={<GetToKnow />} />
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