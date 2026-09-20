import '../styles-css/App.css';


import { Routes, Route } from 'react-router-dom';

import StartPage from "../pages/StartPage.tsx"
import GetToKnow from './GetToKnow.tsx'
import Layout from "../components/Layout.tsx"
import {Login} from "../components/GoogleLogin.tsx"

import AdditionalQuestions from './AdditionalQuestions.tsx';
import Dashboard from './Dashboard.tsx';
import Read from "./Read.tsx"
import Stats from "./Stats.tsx"
import LetterLearning from "./LetterLearning.tsx"
import Quiz from "./Quix.tsx"

import AboutUs from "./AboutUs.tsx"
import QandA from "./QandA.tsx"

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<StartPage />} />

        <Route path="get-started" element={<StartPage />} />
        <Route path="user-information" element={<GetToKnow />} />
        <Route path="login" element={<Login />} />
        <Route path="personalization" element = {<AdditionalQuestions />} />
        <Route path="dashboard" element = {<Dashboard />} />

        <Route path="read" element = {<Read />} />
        <Route path="stats" element = {<Stats />} />
        <Route path = "letter-learning" element={<LetterLearning />} />
        <Route path = "quiz" element={<Quiz />} />

        <Route path="about-us" element = {<AboutUs />} />
        <Route path = "q-and-a" element = {<QandA />} />
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