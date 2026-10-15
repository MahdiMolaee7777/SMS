#!/usr/bin/env node
/**
 * Copies the Chart.js UMD bundle from node_modules into assets/vendor/.
 *
 * The application ships a vendored copy on purpose: the Model Comparison page
 * must keep working without any network access (offline classroom, restricted
 * network, CSP) instead of depending on a public CDN.
 *
 * Usage: npm install && npm run vendor:chart
 */
'use strict';

const fs = require('fs');
const path = require('path');

const projectRoot = path.resolve(__dirname, '..');
const candidates = [
  'node_modules/chart.js/dist/chart.umd.js',
  'node_modules/chart.js/dist/chart.umd.min.js',
];
const destination = path.join(projectRoot, 'assets', 'vendor', 'chart.umd.min.js');

const source = candidates
  .map((candidate) => path.join(projectRoot, candidate))
  .find((candidate) => fs.existsSync(candidate));

if (!source) {
  console.error('Chart.js was not found in node_modules. Run `npm install` first.');
  process.exit(1);
}

fs.mkdirSync(path.dirname(destination), { recursive: true });
fs.copyFileSync(source, destination);

const sizeKb = (fs.statSync(destination).size / 1024).toFixed(1);
console.log(`Vendored ${path.relative(projectRoot, source)} -> ${path.relative(projectRoot, destination)} (${sizeKb} kB)`);
