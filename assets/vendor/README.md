# Vendored front-end libraries

| File | Library | Version | Purpose |
| --- | --- | --- | --- |
| `chart.umd.min.js` | [Chart.js](https://www.chartjs.org/) | 4.4.7 | Bar chart on the Model Comparison page |

These files are committed so that **the application has no runtime dependency on
any external CDN**: it works offline, inside a locked-down network and behind a
strict Content-Security-Policy.

Refresh them with:

```bash
npm install && npm run vendor:chart
```

If Chart.js is ever missing, `assets/js/charts.js` detects it (`typeof window.Chart === 'undefined'`)
and the page falls back to the plain HTML table, which contains the same measured
values.
