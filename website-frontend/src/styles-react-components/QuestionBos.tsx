import type React from "react"
import CustomButton from "./CustomButton";

interface QuestionBoxProps {
    boxText: string;
    onButton1Click: () => void;
    button1Text: string;
    onButton2Click: () => void;
    button2Text: string;
    classNameBox?: string;
    classNameButtons?: string;
    style?: React.CSSProperties;
}

const QuestionBox = ({boxText, onButton1Click, button1Text, onButton2Click, button2Text, classNameBox, classNameButtons, style} : QuestionBoxProps) => {

    return(
        <div className={`flex justify-center items-center ${classNameBox}`} style = {style}>
            <p>{boxText}</p>
            <div className={`flex flex-row items-center justify-between gap-4 ${classNameBox}`}>
                <CustomButton
                    buttonText = {button1Text}
                    onClick = {onButton1Click}
                    className = {classNameButtons}
                    style = {style}
                />
                <CustomButton
                    buttonText = {button2Text}
                    onClick = {onButton2Click}
                    className = {classNameButtons}
                    style = {style}                
                />
                
            </div>
        </div>

    )
}

export default QuestionBox;