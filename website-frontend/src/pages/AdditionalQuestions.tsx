import "../styles-css/App.css";
import QuestionBox from "../styles-react-components/QuestionBos";
import { useNavigate } from "react-router-dom";
import { useState, useEffect} from "react";
import CustomButton from "../styles-react-components/CustomButton"
import { supabase } from "../auths/supabaseClients";

const AdditionalQuestions = () => {
    const [choice, setChoice] = useState<string | null>(null);
    const navigate = useNavigate()

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

    const handleNextClicked = async () => {
        if (!choice) {
            alert("Please select an option.");
            return;
        }

        // Get current authenticated user
        const { data: { user } } = await supabase.auth.getUser();

        if (user) {
            // Determine boolean value for normal_ui column
            const isNormal = choice === "yes";

            // Update normal_ui column in Supabase profiles table
            const { error } = await supabase
                .from("profiles")
                .update({ normal_ui: isNormal })
                .eq("id", user.id);

            if (error) {
                console.error("Failed to update status:", error.message);
                alert("Error updating profile preference.");
            } else {
                console.log("Profile updated successfully!");
                navigate("/dashbaord")
            }

        }
    };

    return (
        <div>
            <QuestionBox
                boxText="Would you like enlarged text or default text size? 
                (Note: if you are hard of vision, we reccommend going with large text.)"
                onButton1Click={handle1Click}
                button1Text="Enlarged Text"
                onButton2Click={handle2Click}
                button2Text="Default Text Size"
            />

            <CustomButton
                buttonText = "Next"
                onClick = {handleNextClicked}

            />
        </div>
    );
};

export default AdditionalQuestions;