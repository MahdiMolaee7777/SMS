/* Global UI behaviour: mobile navigation and small progressive enhancements.
   Everything here is optional - all pages work with JavaScript disabled. */
(function () {
  'use strict';

  // --- Mobile navigation ---------------------------------------------------
  var toggle = document.querySelector('[data-nav-toggle]');
  var links = document.querySelector('[data-nav-links]');

  if (toggle && links) {
    var setOpen = function (open) {
      links.setAttribute('data-open', open ? 'true' : 'false');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    };
    setOpen(false);

    toggle.addEventListener('click', function () {
      setOpen(links.getAttribute('data-open') !== 'true');
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') {
        setOpen(false);
      }
    });

    // Reset the menu state when the layout switches back to desktop.
    window.addEventListener('resize', function () {
      if (window.innerWidth >= 1024) {
        setOpen(false);
      }
    });
  }

  // --- Sample message buttons (detector page) ------------------------------
  var textarea = document.getElementById('id_message');
  var samples = document.querySelectorAll('[data-sample]');
  var counter = document.querySelector('[data-counter-value]');

  var syncCounter = function () {
    if (counter && textarea) {
      counter.textContent = String(textarea.value.length);
    }
  };

  samples.forEach(function (button) {
    button.addEventListener('click', function () {
      if (!textarea) {
        return;
      }
      textarea.value = button.getAttribute('data-sample') || '';
      textarea.focus();
      textarea.dispatchEvent(new Event('input', { bubbles: true }));
      syncCounter();
    });
  });

  if (textarea) {
    textarea.addEventListener('input', syncCounter);
    syncCounter();
  }
})();
