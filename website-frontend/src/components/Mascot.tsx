import { useEffect, useRef, useState, type CSSProperties } from 'react'

// Braillo, Braillie's tutor: a little figure shaped like a braille cell standing up. Its eyes are the top two dots, its belly shows the
// other four (two raised, two flat), and a glowing raised dot sits on its head. It has its own colours, the same in every theme, as a
// character does. It shows what the voice is doing without words: its mouth opens on each word while it speaks, it leans in and cups a
// hand to its ear while it listens, and while it waits it breathes, blinks now and then, and looks about (at the pointer, if there is
// one). It waves hello when it first appears, hops for joy at a right answer (`cheer`), and giggles if it is poked. A picture only: the
// page says the same in words for a screen reader. Everything keeps still for anyone who has asked their device for less motion.

export type MascotState = 'speaking' | 'listening' | 'idle' | 'off'

const still = () => typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
const between = (a: number, b: number) => a + Math.random() * (b - a)
const LOOK_X = 6 // how far the face can look, in the drawing's units
const LOOK_Y = 4

/** A flag that is on for `ms` each time `fire` is called. */
function useMoment(ms: number): [boolean, () => void] {
  const [on, setOn] = useState(false)
  const timer = useRef(0)
  useEffect(() => () => window.clearTimeout(timer.current), [])
  const fire = () => {
    if (still()) return
    window.clearTimeout(timer.current)
    setOn(false)
    window.requestAnimationFrame(() => setOn(true))
    timer.current = window.setTimeout(() => setOn(false), ms)
  }
  return [on, fire]
}

/**
 * `words`: which word of the line is being said (from the captions), so the mouth opens on each new word instead of on a timer.
 * `cheer`: a count that goes up at each right answer; each time it does, Braillo hops for joy.
 */
const Mascot = ({ state, size = 220, words = 0, cheer = 0 }: { state: MascotState; size?: number; words?: number; cheer?: number }) => {
  const box = useRef<SVGSVGElement>(null)
  const [blinking, blink] = useMoment(150)
  const [cheering, celebrate] = useMoment(1100)
  const [giggling, giggle] = useMoment(700)
  const [greeting, greet] = useMoment(1800)
  const [look, setLook] = useState({ x: 0, y: 0 })
  const pointerAt = useRef(0) // when the pointer last moved (it is looked at for a while, then Braillo looks about by himself)

  // hello, once, when he first appears
  useEffect(() => {
    greet()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // a blink every few seconds, at no fixed rhythm, and now and then two in a row
  useEffect(() => {
    if (still()) return
    let timer = 0
    const next = () => {
      timer = window.setTimeout(() => {
        blink()
        if (Math.random() < 0.2) window.setTimeout(blink, 260)
        next()
      }, between(2200, 6000))
    }
    next()
    return () => window.clearTimeout(timer)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // his eyes follow the pointer; with none moving, he glances about by himself now and then
  useEffect(() => {
    if (still()) return
    let frame = 0
    const follow = (e: PointerEvent) => {
      pointerAt.current = Date.now()
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        const r = box.current?.getBoundingClientRect()
        if (!r) return
        const dx = (e.clientX - (r.left + r.width / 2)) / Math.max(r.width, 1)
        const dy = (e.clientY - (r.top + r.height * 0.4)) / Math.max(r.height, 1)
        setLook({ x: Math.max(-1, Math.min(1, dx)) * LOOK_X, y: Math.max(-1, Math.min(1, dy)) * LOOK_Y })
      })
    }
    let glance = 0
    const wander = () => {
      glance = window.setTimeout(() => {
        if (Date.now() - pointerAt.current > 4000) {
          const away = Math.random() < 0.6
          setLook(away ? { x: between(-LOOK_X, LOOK_X), y: between(-LOOK_Y / 2, LOOK_Y) } : { x: 0, y: 0 })
        }
        wander()
      }, between(1800, 4200))
    }
    wander()
    window.addEventListener('pointermove', follow)
    return () => {
      window.removeEventListener('pointermove', follow)
      window.clearTimeout(glance)
      cancelAnimationFrame(frame)
    }
  }, [])

  // a right answer: a hop for joy (only when the count goes up, not on the first look)
  const lastCheer = useRef(cheer)
  useEffect(() => {
    if (cheer > lastCheer.current) celebrate()
    lastCheer.current = cheer
  }, [cheer]) // eslint-disable-line react-hooks/exhaustive-deps

  // while it listens it looks towards the sound; while it talks, mostly at you
  const face = state === 'listening' ? { x: LOOK_X, y: -1 } : state === 'speaking' ? { x: look.x / 3, y: look.y / 3 } : look
  const happy = cheering || giggling
  const classes = ['mascot', blinking && !happy && 'is-blinking', cheering && 'is-cheering', giggling && 'is-giggling', greeting && 'is-greeting']

  return (
    <svg
      ref={box}
      className={classes.filter(Boolean).join(' ')}
      data-state={state}
      viewBox="0 0 220 260"
      width={size}
      height={(size * 260) / 220}
      aria-hidden="true"
      focusable="false"
      onPointerDown={giggle}
      style={{ '--look-x': `${face.x}px`, '--look-y': `${face.y}px` } as CSSProperties}
    >
      {/* a soft shadow on the ground (it shrinks while he is in the air) */}
      <ellipse cx="110" cy="246" rx="58" ry="8" className="mascot__shadow" />

      <g className="mascot__alive">
        <g className="mascot__whole">
          {/* the raised dot on his head, glowing, on a springy stalk */}
          <g className="mascot__antenna">
            <circle cx="110" cy="20" r="17" className="mascot__glow" />
            <path d="M110 44V30" className="mascot__stroke" />
            <circle cx="110" cy="20" r="10" className="mascot__skin mascot__stroke" />
          </g>

          {/* sound coming in, while he listens (beside the hand cupped to his ear, not over it) */}
          <g className="mascot__heard">
            <path d="M204 88c5 6 5 16 0 22" />
            <path d="M214 80c9 11 9 27 0 38" />
          </g>

          {/* arms with mitten hands: the left one rests (and talks with him), the right one goes up to his ear, or waves hello */}
          <g className="mascot__arm mascot__arm--left">
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

          {/* the face, which looks about: brows, eyes (dots 1 and 4) with sparkles, happy eyes for a joyful moment, cheeks, mouth */}
          <g className="mascot__face">
            <path d="M72 84c6-5 14-5 20 0M128 84c6-5 14-5 20 0" className="mascot__stroke mascot__brows" />
            <g className="mascot__eyes">
              <circle cx="84" cy="106" r="12" className="mascot__dark" />
              <circle cx="136" cy="106" r="12" className="mascot__dark" />
              <circle cx="88" cy="101" r="4" className="mascot__sparkle" />
              <circle cx="140" cy="101" r="4" className="mascot__sparkle" />
            </g>
            <path d="M73 110c4-10 18-10 22 0M125 110c4-10 18-10 22 0" className="mascot__stroke mascot__happy" />
            <ellipse cx="68" cy="128" rx="10" ry="6" className="mascot__blush" />
            <ellipse cx="152" cy="128" rx="10" ry="6" className="mascot__blush" />
            <path d="M96 128c8 10 20 10 28 0" className="mascot__stroke mascot__smile" />
            <path d="M92 126c9 16 27 16 36 0z" className="mascot__dark mascot__grin" />
            <g key={words} className="mascot__mouth">
              <path d="M94 126h32c0 14-7 22-16 22s-16-8-16-22z" className="mascot__dark" />
              <ellipse cx="110" cy="142" rx="8" ry="4.5" className="mascot__tongue" />
            </g>
          </g>

          {/* the other four dots of the cell, on his belly: two raised, two flat */}
          <circle cx="88" cy="168" r="8" className="mascot__dark" />
          <circle cx="132" cy="168" r="7" className="mascot__flat" />
          <circle cx="88" cy="196" r="7" className="mascot__flat" />
          <circle cx="132" cy="196" r="8" className="mascot__dark" />

          {/* sparkles that pop out for a right answer */}
          <g className="mascot__sparks">
            <path d="M28 52l3 9 9 3-9 3-3 9-3-9-9-3 9-3z" />
            <path d="M192 40l2.5 7.5 7.5 2.5-7.5 2.5-2.5 7.5-2.5-7.5-7.5-2.5 7.5-2.5z" />
            <path d="M200 150l2 6 6 2-6 2-2 6-2-6-6-2 6-2z" />
          </g>
        </g>
      </g>
    </svg>
  )
}

export default Mascot
