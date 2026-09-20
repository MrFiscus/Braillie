import { getMainStyles } from "./mainStyles";
import { useState } from "react";

interface CustomButtonProps {
  buttonText: string;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
  style?: React.CSSProperties;
  isSelected?: boolean; // Added prop
  type?: "button" | "submit" | "reset"; // Added prop
}

const CustomButton = ({
  buttonText,
  onClick,
  disabled,
  className,
  style,
  isSelected,
  type = "button",
}: CustomButtonProps) => {
  const [scale] = useState<number>(1.0);
  const [isButtonHovered, setIsButtonHovered] = useState<boolean>(false);
  const styles = getMainStyles(scale);

  // Active/selected visual feedback style
  const selectedStyle: React.CSSProperties = isSelected
    ? {
        backgroundColor: "#16a34a", // Green highlight
        color: "#ffffff",
        borderColor: "#15803d",
      }
    : {};

  return (
    <button
      type={type}
      style={{
        ...styles.button,
        ...(isButtonHovered ? styles.buttonHover : {}),
        ...selectedStyle,
        ...style, // Fixed typo from ...{style}
      }}
      onMouseEnter={() => setIsButtonHovered(true)}
      onMouseLeave={() => setIsButtonHovered(false)}
      onClick={onClick}
      disabled={disabled}
      className={className}
    >
      {buttonText}
    </button>
  );
};

export default CustomButton;