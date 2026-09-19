{/* 2. Slider that makes every asset either larger or smaller */}

import { useState } from "react";
import { getMainStyles } from '../styles-react-components/mainStyles.tsx';

/* later reagan problem
interface AccSliderProps {

}
*/
const AccSlider = () => {

  const [scale, setScale] = useState<number>(1.0);
  const styles = getMainStyles(scale)

  return (
    <div style={styles.sliderSection}>
          <div style={styles.sliderHeader}>
            <label htmlFor="asset-scale-slider" style={styles.sliderLabel}>
              Asset Size
            </label>
            <span style={styles.sliderValueBadge}>
              {Math.round(scale * 100)}%
            </span>
          </div>

          <input
            id="asset-scale-slider"
            type="range"
            min="0.75"
            max="1.75"
            step="0.05"
            value={scale}
            onChange={(e) => setScale(parseFloat(e.target.value))}
            style={styles.sliderInput}
            aria-label="Adjust size of all assets on the page"
          />

          <div style={styles.sliderIndicators}>
            <span>Smallest (75%)</span>
            <span>Default (100%)</span>
            <span>Largest (175%)</span>
          </div>
        </div>
  );
}


export default AccSlider;