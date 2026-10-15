/* SMS Detector page: asynchronous form submission with a real loading state.
 *
 * Strategy: the form posts to the normal Django view with fetch(). The response
 * is the fully rendered page, and we swap the server-rendered result fragment
 * (#result-container) and the validation errors. That keeps a single source of
 * truth for markup and escaping - and if anything fails (offline, older browser,
 * JavaScript disabled) the form still submits normally.
 */
(function () {
  'use strict';

  var form = document.getElementById('analyze-form');
  if (!form) {
    return;
  }

  var resultContainer = document.getElementById('result-container');
  var errorContainer = document.getElementById('form-error-container');
  var messageWrapper = document.getElementById('form-message-wrapper');
  var textarea = document.getElementById('id_message');
  var counterValue = document.querySelector('[data-counter-value]');
  var status = document.getElementById('client-status');
  var submitButton = document.getElementById('analyze-button');
  var buttonLabel = submitButton ? submitButton.querySelector('[data-button-label]') : null;
  var spinner = submitButton ? submitButton.querySelector('[data-spinner]') : null;
  var clearButton = document.getElementById('clear-button');
  var modelSelect = document.getElementById('id_model');
  var defaultButtonText = buttonLabel ? buttonLabel.textContent : 'Analyze Message';
  var initiallyDisabled = submitButton ? submitButton.disabled : false;
  var placeholderHtml = resultContainer ? resultContainer.innerHTML : '';
  var inFlight = false;

  function setLoading(loading) {
    inFlight = loading;
    if (spinner) {
      spinner.classList.toggle('hidden', !loading);
    }
    if (buttonLabel) {
      buttonLabel.textContent = loading ? 'Analyzing…' : defaultButtonText;
    }
    if (submitButton) {
      submitButton.disabled = loading || initiallyDisabled;
    }
    if (status) {
      status.textContent = loading
        ? 'Running the selected model on your message…'
        : '';
    }
  }

  function showStatus(message, isError) {
    if (!status) {
      return;
    }
    status.textContent = message || '';
    status.classList.toggle('text-rose-300', Boolean(isError));
    status.classList.toggle('text-slate-400', !isError);
  }

  function updateCounter() {
    if (textarea && counterValue) {
      counterValue.textContent = String(textarea.value.length);
    }
  }

  function clearResultPanel() {
    if (resultContainer) {
      resultContainer.innerHTML = placeholderHtml;
    }
    if (errorContainer) {
      errorContainer.innerHTML = '';
    }
    if (messageWrapper) {
      var existing = messageWrapper.querySelector('#message-errors');
      if (existing) {
        existing.remove();
      }
      if (textarea) {
        textarea.classList.remove('field-invalid');
      }
    }
  }

  function swapFromResponse(html) {
    var parsed = new DOMParser().parseFromString(html, 'text/html');
    var newResult = parsed.getElementById('result-container');
    var newErrors = parsed.getElementById('form-error-container');
    var newMessageErrors = parsed.querySelector('#form-message-wrapper');

    if (resultContainer && newResult) {
      resultContainer.innerHTML = newResult.innerHTML;
    }
    if (errorContainer && newErrors) {
      errorContainer.innerHTML = newErrors.innerHTML;
    }
    if (messageWrapper && newMessageErrors) {
      messageWrapper.innerHTML = newMessageErrors.innerHTML;
      var hasError = Boolean(messageWrapper.querySelector('.field-error'));
      if (textarea) {
        textarea.classList.toggle('field-invalid', hasError);
      }
    }

    var sharedError = parsed.querySelector('.notice-danger');
    return Boolean(sharedError && sharedError.textContent.trim());
  }

  form.addEventListener('submit', function (event) {
    if (inFlight) {
      return;
    }
    if (!window.fetch || !window.DOMParser) {
      return; // let the browser perform a normal POST
    }
    // Let the browser handle obvious client-side issues (e.g. required field).
    if (!form.checkValidity()) {
      return;
    }

    event.preventDefault();
    setLoading(true);
    showStatus('', false);
    clearResultPanel();

    var submitUrl = form.getAttribute('action') || window.location.pathname;
    window
      .fetch(submitUrl, {
        method: 'POST',
        body: new FormData(form),
        headers: {
          'X-Requested-With': 'XMLHttpRequest',
          'X-CSRFToken': getCookie('csrftoken') || '',
        },
        credentials: 'same-origin',
      })
      .then(function (response) {
        return response.text().then(function (html) {
          return { ok: response.ok, html: html };
        });
      })
      .then(function (payload) {
        var hadError = swapFromResponse(payload.html);
        if (!payload.ok) {
          showStatus('The server rejected the request. Please check the highlighted fields.', true);
        } else if (hadError) {
          showStatus('The analysis could not be completed. See the message on the left.', true);
        } else {
          showStatus('Analysis complete.', false);
        }
        if (resultContainer && window.innerWidth < 1024) {
          resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      })
      .catch(function () {
        showStatus(
          'Network error: the analysis was not sent. Check your connection or press the button again.',
          true
        );
      })
      .finally(function () {
        setLoading(false);
      });
  });

  if (clearButton) {
    clearButton.addEventListener('click', function () {
      form.reset();
      if (textarea) {
        textarea.value = '';
        textarea.focus();
      }
      clearResultPanel();
      updateCounter();
      showStatus('Input and result cleared.', false);
    });
  }

  if (textarea) {
    textarea.addEventListener('input', updateCounter);
    updateCounter();
  }

  if (modelSelect) {
    var toggleModelNotes = function () {
      document.querySelectorAll('[data-model-note]').forEach(function (note) {
        note.classList.toggle('hidden', note.getAttribute('data-model-note') !== modelSelect.value);
      });
    };
    modelSelect.addEventListener('change', toggleModelNotes);
    toggleModelNotes();
  }

  function getCookie(name) {
    var match = document.cookie.match(new RegExp('(^|;\\s*)' + name + '=([^;]*)'));
    return match ? decodeURIComponent(match[2]) : '';
  }
})();
