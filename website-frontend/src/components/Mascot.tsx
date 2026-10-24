// Braillo, Braillie's tutor: a little figure shaped like a braille cell standing up. Its eyes are the top two dots, its belly shows the
// other four (two raised, two flat), and a glowing raised dot sits on its head. It has its own colours, the same in every theme, as a
// character does. It shows what the voice is doing without words: its mouth opens on each word while it speaks, it cups a hand to its
// ear while it listens, and it smiles and blinks while it waits. A picture only: the page says the same in words for a screen reader.

export type MascotState = 'speaking' | 'listening' | 'idle' | 'off'

/** `words`: which word of the line is being said (from the captions), so the mouth opens on each new word instead of on a timer. */
const Mascot = ({ state, size = 220, words = 0 }: { state: MascotState; size?: number; words?: number }) => (
  <svg className="mascot" data-state={state} viewBox="0 0 220 260" width={size} height={(size * 260) / 220} aria-hidden="true" focusable="false">
    {/* a soft shadow on the ground */}
    <ellipse cx="110" cy="246" rx="58" ry="8" className="mascot__shadow" />

    <g className="mascot__whole">
      {/* the raised dot on its head, glowing */}
      <circle cx="110" cy="20" r="17" className="mascot__glow" />
      <path d="M110 44V30" className="mascot__stroke" />
      <circle cx="110" cy="20" r="10" className="mascot__skin mascot__stroke" />

      {/* sound coming in, while it listens (beside the hand cupped to its ear, not over it) */}
      <g className="mascot__heard">
        <path d="M204 88c5 6 5 16 0 22" />
        <path d="M214 80c9 11 9 27 0 38" />
      </g>

      {/* arms with mitten hands: the left one rests, the right one goes up to its ear to listen */}
      <g className="mascot__arm">
        <path d="M52 128c-16 8-24 24-22 40" className="mascot__limb" />
        <path d="M52 128c-16 8-24 24-22 40" className="mascot__limb mascot__limb--in" />
        <circle cx="31" cy="171" r="11" className="mascot__skin mascot__stroke" />
      </g>
      <g className="mascot__arm mascot__arm--down">
        <path d="M168 128c16 8 24 24 22 40" className="mascot__limb" />
        <path d="M168 128c16 8 24 24 22 40" className="mascot__limb mascot__limb--in" />
        <circle cx="189" cy="171" r="11" className="mascot__skin mascot__stroke" />
      </g>
      <g className="mascot__arm mascot__arm--up">
        <path d="M168 128c12-4 18-14 18-26" className="mascot__limb" />
        <path d="M168 128c12-4 18-14 18-26" className="mascot__limb mascot__limb--in" />
        <circle cx="187" cy="99" r="10" className="mascot__skin mascot__stroke" />
      </g>

      {/* feet */}
      <rect x="72" y="214" width="30" height="20" rx="10" className="mascot__skin mascot__stroke" />
      <rect x="118" y="214" width="30" height="20" rx="10" className="mascot__skin mascot__stroke" />

      {/* the body: a braille cell, rounded like a pebble, with a shine */}
      <rect x="44" y="40" width="132" height="184" rx="62" className="mascot__skin mascot__stroke mascot__body" />
      <ellipse cx="78" cy="78" rx="18" ry="10" transform="rotate(-30 78 78)" className="mascot__shine" />

      {/* brows and eyes (dots 1 and 4), each eye with a sparkle */}
      <path d="M72 84c6-5 14-5 20 0M128 84c6-5 14-5 20 0" className="mascot__stroke mascot__brows" />
      <g className="mascot__eyes">
        <circle cx="84" cy="106" r="12" className="mascot__dark" />
        <circle cx="136" cy="106" r="12" className="mascot__dark" />
        <circle cx="88" cy="101" r="4" className="mascot__sparkle" />
        <circle cx="140" cy="101" r="4" className="mascot__sparkle" />
      </g>
      <ellipse cx="68" cy="128" rx="10" ry="6" className="mascot__blush" />
      <ellipse cx="152" cy="128" rx="10" ry="6" className="mascot__blush" />

      {/* the mouth: a smile while it waits, open (with a tongue) while it talks */}
      <path d="M96 128c8 10 20 10 28 0" className="mascot__stroke mascot__smile" />
      <g key={words} className="mascot__mouth">
        <path d="M94 126h32c0 14-7 22-16 22s-16-8-16-22z" className="mascot__dark" />
        <ellipse cx="110" cy="142" rx="8" ry="4.5" className="mascot__tongue" />
      </g>

      {/* the other four dots of the cell, on its belly: two raised, two flat */}
      <circle cx="88" cy="168" r="8" className="mascot__dark" />
      <circle cx="132" cy="168" r="7" className="mascot__flat" />
      <circle cx="88" cy="196" r="7" className="mascot__flat" />
      <circle cx="132" cy="196" r="8" className="mascot__dark" />
    </g>
  </svg>
)

export default Mascot
