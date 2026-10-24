import { useEffect, type ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import AccountKey from '../components/AccountKey.tsx'
import DisplayMenu from '../components/DisplayMenu.tsx'
import SiteFooter from '../components/SiteFooter.tsx'

const UPDATED = '24 October 2026'
const ISSUES = 'https://github.com/MrFiscus/Braillie/issues'

/** A plain page of text: the landing page's header, a title, the sections, and the footer. Written to be read aloud: short sections,
 * real headings, no tables. */
const LegalPage = ({ title, intro, children }: { title: string; intro: string; children: ReactNode }) => {
  const { hash } = useLocation()
  useEffect(() => {
    document.title = `${title} – Braillie`
    if (hash) document.getElementById(hash.slice(1))?.scrollIntoView()
    else window.scrollTo(0, 0)
  }, [title, hash])

  return (
    <div className="landing legal-page" id="top">
      <a className="skip" href="#main">
        Skip to the page
      </a>
      <header className="landing__head">
        <Link className="brand" to="/" aria-label="Braillie, home">
          <span>
            Braillie<span className="brand__dot">.</span>
          </span>
        </Link>
        <div className="landing__nav">
          <AccountKey />
          <DisplayMenu share={false} />
        </div>
      </header>
      <main id="main" tabIndex={-1} className="legal">
        <p className="legal__kicker">Last updated {UPDATED}</p>
        <h1>{title}</h1>
        <p className="legal__intro">{intro}</p>
        {children}
      </main>
      <SiteFooter />
    </div>
  )
}

const Contact = () => (
  <p>
    Questions, requests or problems: open an issue on{' '}
    <a href={ISSUES} target="_blank" rel="noreferrer">
      our GitHub page<span className="sr-only"> (opens in a new tab)</span>
    </a>
    . We read every one.
  </p>
)

export const Privacy = () => (
  <LegalPage
    title="Privacy policy"
    intro="Braillie is built to collect as little as possible. This page says what it uses, where it goes, and how to remove it."
  >
    <section aria-labelledby="p-short">
      <h2 id="p-short">The short version</h2>
      <ul>
        <li>Camera pictures stay on the computer running Braillie. They are never sent to any other company.</li>
        <li>What you say is sent to Deepgram, only to turn your speech into words and Braillie’s words into speech.</li>
        <li>Braillo’s answers come from Claude, by Anthropic. It gets lesson facts and your question, never pictures, audio or your name.</li>
        <li>No adverts, no tracking, and we never sell anything about you.</li>
      </ul>
    </section>

    <section aria-labelledby="p-what">
      <h2 id="p-what">What Braillie uses</h2>
      <h3>Your camera</h3>
      <p>
        Your phone’s camera (or the laptop’s) watches the printed sheet and your fingertip. The pictures go over your own Wi-Fi to the computer
        running Braillie, are used there to find the braille and your finger, and are not saved or sent anywhere else.
      </p>
      <h3>Your voice</h3>
      <p>
        While the microphone is on, the sound is streamed to Deepgram to recognise what you say, and Braillie’s spoken lines are made by
        Deepgram too. Press the microphone key to mute it: nothing is sent while it is muted.
      </p>
      <h3>Questions to Braillo</h3>
      <p>
        When you ask Braillo something, the words of your question and facts about the lesson (the letter, its dots, your score) are sent to
        Claude, made by Anthropic, to write the answer. Your name, pictures and audio are never included.
      </p>
      <h3>Your account</h3>
      <p>
        If you sign in with Google, we receive your name and email address from Google, and your progress (which letters you know, your
        lessons and streaks) is kept in your account so you can carry on from another computer. Only you can read it. As a guest, only the
        first name you type is used, for the greeting, and it is forgotten when you close the tab.
      </p>
    </section>

    <section aria-labelledby="storage">
      <h2 id="storage">Cookies and storage</h2>
      <p>
        Braillie does not use advertising or tracking cookies. Your browser keeps a few small things for Braillie itself: your display
        choices (text size and colours), a guest’s first name for the current tab, and, if you sign in, the sign-in itself. Clearing your
        browser’s data for this site removes them.
      </p>
    </section>

    <section aria-labelledby="p-who">
      <h2 id="p-who">Who else is involved</h2>
      <ul>
        <li>Deepgram: speech recognition and the voice.</li>
        <li>Anthropic (Claude): Braillo’s answers and memory aids.</li>
        <li>Supabase: sign-in and keeping a signed-in learner’s progress.</li>
        <li>Google: signing in with your Google account, if you choose to.</li>
      </ul>
      <p>Each handles the data under its own privacy policy, only to provide that part of Braillie.</p>
    </section>

    <section aria-labelledby="p-rights">
      <h2 id="p-rights">Your choices</h2>
      <ul>
        <li>Use Braillie as a guest: nothing is kept in an account.</li>
        <li>Mute the microphone at any time.</li>
        <li>Ask us to delete your account and progress, or for a copy of what we hold, using the contact below.</li>
      </ul>
    </section>

    <section aria-labelledby="p-children">
      <h2 id="p-children">Children</h2>
      <p>
        Braillie is for learners of all ages. A child under 13 should use it as a guest, or with a parent or teacher who signs in for them.
      </p>
    </section>

    <section aria-labelledby="p-changes">
      <h2 id="p-changes">Changes and contact</h2>
      <p>If this policy changes, the date at the top will change too.</p>
      <Contact />
    </section>
  </LegalPage>
)

export const Terms = () => (
  <LegalPage title="Terms of use" intro="Using Braillie means agreeing to these terms. They are short, and written to be read aloud.">
    <section aria-labelledby="t-what">
      <h2 id="t-what">What Braillie is</h2>
      <p>
        Braillie is a free learning tool, built at HackMIT, that helps people practise reading braille by touch. It is a project in
        development, not a replacement for a qualified braille teacher, and it can make mistakes: it may misread a dot or a word.
      </p>
    </section>

    <section aria-labelledby="t-use">
      <h2 id="t-use">Using it well</h2>
      <ul>
        <li>Use Braillie to learn, for yourself or to help someone else learn.</li>
        <li>Do not point the camera at people or private documents that are not yours to share.</li>
        <li>Do not try to break, overload or misuse Braillie or the services it relies on.</li>
      </ul>
    </section>

    <section aria-labelledby="t-account">
      <h2 id="t-account">Your account</h2>
      <p>
        You can use Braillie without an account. If you sign in, keep your Google account secure; you can stop using Braillie and ask us to
        delete your account at any time.
      </p>
    </section>

    <section aria-labelledby="t-services">
      <h2 id="t-services">Other services</h2>
      <p>
        Braillie relies on Deepgram, Anthropic, Supabase and Google. If one of them is unavailable, parts of Braillie may stop working for a
        while. Their own terms apply to them. How data is shared with them is explained in the{' '}
        <Link to="/privacy">privacy policy</Link>.
      </p>
    </section>

    <section aria-labelledby="t-asis">
      <h2 id="t-asis">No guarantees</h2>
      <p>
        Braillie is provided as it is, without any promise that it will always work or always be right. As far as the law allows, we are not
        responsible for any loss that comes from using it.
      </p>
    </section>

    <section aria-labelledby="t-code">
      <h2 id="t-code">The code</h2>
      <p>Braillie’s source code is on GitHub, where its licence says how the code itself may be used.</p>
    </section>

    <section aria-labelledby="t-changes">
      <h2 id="t-changes">Changes and contact</h2>
      <p>We may update these terms as Braillie grows. The date at the top shows the latest version.</p>
      <Contact />
    </section>
  </LegalPage>
)

export const Accessibility = () => (
  <LegalPage
    title="Accessibility"
    intro="Braillie is made for people who are blind or have low vision, so it is designed to be used without seeing the screen."
  >
    <section aria-labelledby="a-how">
      <h2 id="a-how">How Braillie is built to be used</h2>
      <ul>
        <li>Everything can be done by voice: say “learn”, “hint”, “repeat”, or ask Braillo a question.</li>
        <li>Everything can also be done with a keyboard, a screen reader, or a mouse.</li>
        <li>Pages use real headings, labels and landmarks, and announce what changes.</li>
        <li>Text is large and clear, with four text sizes, and colours meet the strictest contrast level (WCAG AAA, 7 to 1).</li>
        <li>Black on white and yellow on black themes are used automatically when your device asks for more contrast.</li>
        <li>Animations stop when your device asks for less motion.</li>
        <li>The phone page uses the same colours as the laptop, and speaks its own status out loud.</li>
      </ul>
    </section>

    <section aria-labelledby="a-standard">
      <h2 id="a-standard">Our aim</h2>
      <p>
        We aim to meet the Web Content Accessibility Guidelines (WCAG) 2.2 at level AA, and AAA for colour contrast, which is checked
        automatically every time Braillie is built. Pages are checked with the keyboard alone and at the largest text sizes.
      </p>
    </section>

    <section aria-labelledby="a-known">
      <h2 id="a-known">Known limits</h2>
      <ul>
        <li>The camera needs enough light and the whole sheet in view to read it.</li>
        <li>Speech recognition can mishear in a noisy room. Every voice action also has a key.</li>
      </ul>
    </section>

    <section aria-labelledby="a-tell">
      <h2 id="a-tell">Tell us what does not work</h2>
      <p>If anything in Braillie is hard to use with your setup, we want to know and will fix it.</p>
      <Contact />
    </section>
  </LegalPage>
)
