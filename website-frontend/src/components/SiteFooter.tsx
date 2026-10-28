import { Link } from 'react-router-dom'

const REPO = 'https://github.com/learnwithbraillie/Braillie'

// (plain <a> for the landing page's sections: they are on another page from the legal pages, and a full load lands on the right spot)
const COLUMNS: { title: string; links: { label: string; to: string; external?: boolean; page?: boolean }[] }[] = [
  {
    title: 'Braillie',
    links: [
      { label: 'How it works', to: '/#how', page: true },
      { label: 'Practice sheets', to: '/#sheets', page: true },
      { label: 'Meet Braillo', to: '/#braillo', page: true },
      { label: 'Who it is for', to: '/#made-for', page: true },
      { label: 'Start learning', to: '/login' },
    ],
  },
  {
    title: 'Help',
    links: [
      { label: 'Accessibility', to: '/accessibility' },
      { label: 'Contact us', to: `${REPO}/issues`, external: true },
      { label: 'Source code', to: REPO, external: true },
    ],
  },
  {
    title: 'Legal',
    links: [
      { label: 'Privacy policy', to: '/privacy' },
      { label: 'Terms of use', to: '/terms' },
      { label: 'Cookies and storage', to: '/privacy#storage', page: true },
    ],
  },
]

/** The foot of the public pages: what Braillie is, the ways around, help, the legal pages, and who made it. */
const SiteFooter = () => (
  <footer className="sitefoot">
    <div className="sitefoot__top">
      <div className="sitefoot__brand">
        <p className="sitefoot__name">
          Braillie<span className="brand__dot">.</span>
        </p>
        <p>Learn braille by touch, with a patient voice and your phone’s camera.</p>
      </div>
      {COLUMNS.map((c) => (
        <nav key={c.title} className="sitefoot__col" aria-label={c.title}>
          <h2>{c.title}</h2>
          <ul>
            {c.links.map((l) => (
              <li key={l.label}>
                {l.external ? (
                  <a href={l.to} target="_blank" rel="noreferrer">
                    {l.label}
                    <span className="sr-only"> (opens in a new tab)</span>
                  </a>
                ) : l.page ? (
                  <a href={l.to}>{l.label}</a>
                ) : (
                  <Link to={l.to}>{l.label}</Link>
                )}
              </li>
            ))}
          </ul>
        </nav>
      ))}
    </div>
    <div className="sitefoot__bottom">
      <p>© {new Date().getFullYear()} Braillie. Built at HackMIT.</p>
      <a href="#top">Back to top ↑</a>
    </div>
  </footer>
)

export default SiteFooter
