/* Model Comparison page: grouped bar chart of the evaluation metrics.
 *
 * The series are rendered server-side into a JSON <script> tag (Django's
 * json_script filter), so the page still shows the numbers in an HTML table when
 * Chart.js cannot be loaded.
 */
(function () {
  'use strict';

  var canvas = document.getElementById('metrics-chart');
  var dataElement = document.getElementById('chart-data');
  var fallbackMessage = document.querySelector('[data-chart-fallback]');

  if (!canvas || !dataElement) {
    return;
  }

  function showFallback(reason) {
    canvas.parentElement.innerHTML =
      '<p class="hint">' +
      'The chart library is unavailable (' +
      reason +
      '), so the table below shows the same measured values.</p>';
    if (fallbackMessage) {
      fallbackMessage.classList.remove('hidden');
    }
  }

  if (typeof window.Chart === 'undefined') {
    showFallback('assets not loaded');
    return;
  }

  var payload;
  try {
    payload = JSON.parse(dataElement.textContent);
  } catch (error) {
    showFallback('invalid data');
    return;
  }

  var palette = [
    { background: 'rgba(56, 189, 248, 0.85)', border: 'rgba(56, 189, 248, 1)' },
    { background: 'rgba(139, 92, 246, 0.85)', border: 'rgba(139, 92, 246, 1)' },
    { background: 'rgba(251, 146, 60, 0.85)', border: 'rgba(251, 146, 60, 1)' },
    { background: 'rgba(52, 211, 153, 0.85)', border: 'rgba(52, 211, 153, 1)' },
  ];

  var datasets = (payload.datasets || []).map(function (dataset, index) {
    var colors = palette[index % palette.length];
    return {
      label: dataset.label,
      data: dataset.data,
      backgroundColor: colors.background,
      borderColor: colors.border,
      borderWidth: 1,
      borderRadius: 6,
      maxBarThickness: 46,
    };
  });

  new window.Chart(canvas, {
    type: 'bar',
    data: { labels: payload.labels || [], datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 500, easing: 'easeOutQuart' },
      layout: { padding: { top: 8 } },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { size: 11 } },
        },
        y: {
          beginAtZero: true,
          suggestedMax: 100,
          grid: { color: 'rgba(148, 163, 184, 0.12)' },
          ticks: {
            color: '#94a3b8',
            font: { size: 11 },
            callback: function (value) {
              return value + '%';
            },
          },
        },
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#cbd5e1', boxWidth: 12, usePointStyle: true, padding: 16 },
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              return context.dataset.label + ': ' + context.parsed.y + '%';
            },
          },
        },
      },
    },
  });
})();
