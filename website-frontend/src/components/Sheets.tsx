// The practice sheets, laid out like "Meet Braillo": on one side what to do (the heading, one line on making a sheet, and one key that
// downloads all three, an A4 PDF of the "poke side", a page each); on the other, the three pages fanned out like paper on a desk.

const PAGES = ['alphabet', 'words', 'lookalikes']

const Sheets = () => (
  <div className="sheets">
    <div className="sheets__copy">
      <h2 id="sheets-h">Print your sheets</h2>
      <p>Print at 100% on A4. Lay each page face down on something soft, press every dot through, and turn it over to feel.</p>
      <a className="cta sheets__get" href="/sheets/braillie-sheets.pdf" download>
        <svg className="btn__icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <path d="M12 4v11m-5-5 5 5 5-5M5 20h14" />
        </svg>
        Download all sheets<span className="sr-only">: the alphabet, words and look-alikes sheets, one PDF to print on A4</span>
      </a>
    </div>
    <div className="sheets__fan" aria-hidden="true">
      {PAGES.map((p) => (
        <img key={p} className={`sheets__page sheets__page--${p}`} src={`/sheets/page-${p}.png`} alt="" width={520} height={735} loading="lazy" />
      ))}
    </div>
  </div>
)

export default Sheets
