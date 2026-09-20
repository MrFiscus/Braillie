import "../styles-css/App.css";
import QuestionBox from "../styles-react-components/QuestionBos";
//import { useNavigate } from "react-router-dom";
import { useState, useEffect} from "react";
import CustomButton from "../styles-react-components/CustomButton"
import { supabase } from "../auths/supabaseClients";

const AdditionalQuestions = () => {
    const [choice, setChoice] = useState<string | null>(null);


    useEffect(() => {
    async function loadUserProfile() {
      // 1. Get current logged-in user
      const { data: { user } } = await supabase.auth.getUser();

      if (user) {
        // 2. Fetch the is_cool status from public.profiles
        const { data, error } = await supabase
          .from("profiles")
          .select("choice")
          .eq("id", user.id)
          .single();

        if (data && !error) {
          setChoice(data.choice ?? null);
        }
      }
    }

    loadUserProfile();
  }, []);
    const handle1Click = () => {
        setChoice("yes");
    };

    const handle2Click = () => {
        setChoice("no");
    };

    const handleNextClicked = () => {
        if (choice === "no") {
            // handle no choice
        } else if (choice === "yes") {
            // handle yes choice
        } else {
            alert("Please select an option.");
        }
    };

    return (
        <div>
            <QuestionBox
                boxText=""
                onButton1Click={handle1Click}
                button1Text="Im not GAYYY!!!!"
                onButton2Click={handle2Click}
                button2Text="im gay (sadly true im so sorry)"
            />

            <CustomButton
                buttonText = "Next"
                onClick = {handleNextClicked}

            />
        </div>
    );
};

export default AdditionalQuestions;