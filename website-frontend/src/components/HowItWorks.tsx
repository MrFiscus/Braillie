// How Braillie works, as three pictures instead of paragraphs: print and press a sheet, point the phone at it, touch a letter and hear
// it. Each scene moves a little to show what happens (a sheet coming out, a pen pressing a dot, the camera sweeping the page, a finger on
// a cell while the voice speaks); everything keeps still when the device asks for less motion. For a screen reader each
// picture is described in full, since it cannot see the scene.

const DOT = (cx: number, cy: number, on: boolean, cls = '') => (
  <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r={on ? 7 : 4.5} className={`how__dot${on ? ' is-on' : ''}${cls ? ` ${cls}` : ''}`} />
)

/** 1. A printer feeds out a practice sheet; on a soft mat, a pen presses each dot through. */
const PrintScene = () => (
  <svg viewBox="0 0 320 220" className="how__scene" aria-hidden="true" focusable="false">
    {/* the printer */}
    <rect x="18" y="40" width="112" height="22" rx="6" className="how__line how__soft" />
    <rect x="8" y="58" width="132" height="58" rx="12" className="how__line how__fill" />
    <circle cx="118" cy="78" r="5" className="how__accent-fill" />
    {/* the sheet coming out */}
    <g className="how__feed">
      <rect x="30" y="102" width="88" height="70" rx="4" className="how__line how__paper" />
      {DOT(52, 122, true)}
      {DOT(52, 140, true)}
      {DOT(52, 158, false)}
      {DOT(70, 122, false)}
      {DOT(70, 140, true)}
      {DOT(70, 158, false)}
      {DOT(96, 122, true)}
      {DOT(96, 140, false)}
      {DOT(96, 158, false)}
    </g>
    {/* the arrow to the next part */}
    <path d="M150 112h22m-8-8 8 8-8 8" className="how__line how__arrow" />
    {/* the soft mat, the sheet face down, and the pen */}
    <rect x="184" y="150" width="128" height="34" rx="12" className="how__soft-fill" />
    <rect x="196" y="122" width="104" height="40" rx="4" className="how__line how__paper" />
    {DOT(220, 134, true)}
    {DOT(220, 150, false)}
    {DOT(240, 134, true)}
    {DOT(260, 134, false)}
    {DOT(260, 150, true, 'how__pressed')}
    {DOT(280, 134, false)}
    <g className="how__pen">
      <path d="M260 120 248 64l14-4 12 56z" className="how__line how__accent-fill" />
      <path d="M260 120l-2-10 8-2z" className="how__ink-fill" />
    </g>
  </svg>
)

/** 2. The laptop shows a code, the phone reads it, and the phone's camera then watches the sheet on the desk. */
const PhoneScene = () => (
  <svg viewBox="0 0 320 220" className="how__scene" aria-hidden="true" focusable="false">
    {/* the laptop, with the code */}
    <rect x="10" y="44" width="104" height="70" rx="8" className="how__line how__fill" />
    <path d="M0 122h124l-10 12H10z" className="how__line how__soft" />
    <g className="how__ink-fill">
      <rect x="38" y="56" width="16" height="16" rx="2" />
      <rect x="70" y="56" width="16" height="16" rx="2" />
      <rect x="38" y="86" width="16" height="16" rx="2" />
      <rect x="60" y="78" width="7" height="7" />
      <rect x="72" y="88" width="7" height="7" />
      <rect x="80" y="78" width="7" height="7" />
    </g>
    {/* the link travelling to the phone */}
    <path d="M118 78c30-30 60-34 92-22" className="how__line how__link" />
    {/* the phone, held over the desk */}
    <g className="how__phone">
      <rect x="196" y="20" width="62" height="108" rx="12" className="how__line how__fill" />
      <circle cx="227" cy="32" r="4" className="how__ink-fill" />
      <rect x="206" y="42" width="42" height="70" rx="6" className="how__screen" />
      <path d="M216 78l8 8 16-18" className="how__check" />
    </g>
    {/* what its camera sees: the sheet on the desk */}
    <path d="M206 128 150 200h154l-56-72z" className="how__beam" />
    <rect x="160" y="180" width="134" height="30" rx="4" className="how__line how__paper" />
    {DOT(184, 195, true)}
    {DOT(204, 195, true)}
    {DOT(228, 195, false)}
    {DOT(250, 195, true)}
    {DOT(272, 195, false)}
  </svg>
)

/** 3. A finger rests on a braille cell, and the voice says which letter it is. */
const TouchScene = () => (
  <svg viewBox="0 0 320 220" className="how__scene" aria-hidden="true" focusable="false">
    {/* the cell: the letter D, dots 1, 4 and 5 */}
    <rect x="24" y="74" width="96" height="130" rx="18" className="how__line how__paper" />
    {DOT(54, 104, true)}
    {DOT(54, 139, false)}
    {DOT(54, 174, false)}
    {DOT(90, 104, true)}
    {DOT(90, 139, true)}
    {DOT(90, 174, false)}
    {/* the fingertip, resting on it */}
    <g className="how__finger">
      <path d="M64 92c-2-30 4-80 8-92h40c2 14 4 60 0 92-2 16-46 16-48 0z" className="how__line how__skin" />
      <path d="M74 72c0-10 4-16 14-16s14 6 14 16" className="how__line how__nail" />
      <path d="M76 22c8 3 18 3 26 0M76 34c8 3 18 3 26 0" className="how__line how__nail" />
    </g>
    {/* the voice */}
    <g className="how__waves">
      <path d="M150 118c8 8 8 24 0 32" className="how__line how__wave how__wave--1" />
      <path d="M164 106c14 16 14 40 0 56" className="how__line how__wave how__wave--2" />
      <path d="M178 94c20 22 20 58 0 80" className="how__line how__wave how__wave--3" />
    </g>
    <path d="M208 70h88a14 14 0 0 1 14 14v70a14 14 0 0 1-14 14h-52l-24 22v-22h-12a14 14 0 0 1-14-14V84a14 14 0 0 1 14-14z" className="how__line how__fill" />
    <text x="252" y="140" textAnchor="middle" className="how__letter">
      D
    </text>
  </svg>
)

const STEPS = [
  {
    label: 'Print and press',
    Scene: PrintScene,
    describe:
      'Step 1: print one of the practice sheets, lay it face down on something soft, and press each dot through with a blunt pen. The bumps on the front are what you feel.',
  },
  {
    label: 'Scan with your phone',
    Scene: PhoneScene,
    describe:
      'Step 2: scan the code on the laptop screen with your phone. The phone then watches the sheet and your fingertip, and lets Braillie hear you and speak to you.',
  },
  {
    label: 'Touch and listen',
    Scene: TouchScene,
    describe:
      'Step 3: rest a finger on a letter, and Braillie tells you which letter it is. It teaches a few letters at a time, quizzes you, and reads words aloud.',
  },
]

const HowItWorks = () => (
  <ol className="how">
    {STEPS.map(({ label, Scene, describe }, n) => (
      <li key={label} className="how__step">
        <figure role="img" aria-label={describe}>
          <Scene />
          <figcaption aria-hidden="true">{label}</figcaption>
        </figure>
        {n < STEPS.length - 1 && (
          <svg className="how__next" viewBox="0 0 48 24" aria-hidden="true" focusable="false">
            <path d="M2 12h40m-10-10 10 10-10 10" />
          </svg>
        )}
      </li>
    ))}
  </ol>
)

export default HowItWorks
