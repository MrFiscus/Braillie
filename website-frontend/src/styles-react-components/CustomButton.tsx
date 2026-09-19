import { getMainStyles } from "./mainStyles";
import {useState} from "react";


interface CustomButtonProps {
    buttonText: string;
    onClick: () => void;
    disabled?: boolean;
    className?: string;
    style?: React.CSSProperties;
}

const CustomButton = ({buttonText, onClick, disabled, className, style} : CustomButtonProps) => {
    const [scale] = useState<number>(1.0)
    const [isButtonHovered, setIsButtonHovered] = useState<boolean>(false)


    const styles = getMainStyles(scale)

    return(
        <button
            type="button"
            style={{
            ...styles.button,
            ...(isButtonHovered ? styles.buttonHover : {}), ...{style}
            }}
            onMouseEnter={() => setIsButtonHovered(true)}
            onMouseLeave={() => setIsButtonHovered(false)}
            onClick={onClick}
            disabled = {disabled}
            className = {className}
        >
            {buttonText}
        </button>
    );
}

export default CustomButton