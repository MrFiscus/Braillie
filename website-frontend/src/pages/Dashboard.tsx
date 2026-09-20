import CustomButton from "../styles-react-components/CustomButton"
import {useNavigate} from "react-router-dom"


const Dashboard = () => {
    const navigate = useNavigate()

    const handleReadClick = () => {
        navigate("/read")
    }

    const handleAZClick = () => {
        navigate("/letter-learning")
    }

    const handleQuizClick = () => {
        navigate("/quiz")
    }

    const handleStatsClick = () => {
        navigate("/stats")
    }

    return(
        <div className="flex items-center">
            <title>Dashboard</title>

            <h2>Learning Dashboard</h2>

            <div className = {`flex flex-row items-center justify-between gap-4`}>
                <CustomButton
                    buttonText = "Read"
                    onClick = {handleReadClick}
                />

                <CustomButton
                    buttonText = "Letter Learn"
                    onClick = {handleAZClick}
                />

                <CustomButton
                    buttonText = "Quiz"
                    onClick = {handleQuizClick}
                />
            </div>
            <CustomButton 
                buttonText = "Your Learning Statistics"
                onClick = {handleStatsClick}
            />
        </div>

    );
}

export default Dashboard