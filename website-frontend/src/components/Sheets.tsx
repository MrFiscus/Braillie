// The practice sheets: on one side what to do (the heading, four short steps to make a sheet, and one key that downloads all four, an A4
// PDF printed on both sides: the reading side with black dots, the poking side mirrored, with a circle round each dot to push
// through); on the other, the four pages, which fan out like paper on a desk as they scroll into view, the one you point at lifting
// forward.

const PAGES = ['alphabet', 'words', 'lookalikes', 'numbers']

const Sheets = () => (
  <div className="sheets">
    <div className="sheets__copy">
      <h2 id="sheets-h">Print your sheets</h2>
      <ol className="sheets__points">
        <li>Print both sides, at 100% on A4.</li>
        <li>Lay it circles up on something soft.</li>
        <li>Push through every circle.</li>
        <li>Turn it over and feel.</li>
      </ol>
      <a className="cta sheets__get" href="/sheets/braillie-sheets.pdf" download>
        <svg className="btn__icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <path d="M12 4v11m-5-5 5 5 5-5M5 20h14" />
        </svg>
        Download all sheets<span className="sr-only">: the alphabet, words, look-alikes and numbers sheets, one PDF to print on both sides of A4</span>
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
