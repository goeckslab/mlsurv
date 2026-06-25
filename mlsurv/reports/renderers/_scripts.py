"""JavaScript constants for the HTML report renderer."""


def wrap_script(js: str) -> str:
    """Wrap raw JavaScript in ``<script>`` tags."""
    return f"<script>\n{js}\n</script>"


TABS_SCRIPT: str = """\
(function () {
  document.addEventListener('click', function (event) {
    const button = event.target.closest('.tab-button');
    if (!button) return;

    const group = button.getAttribute('data-tab-group');
    const target = button.getAttribute('data-tab-target');
    if (!group || !target) return;

    document.querySelectorAll('.tab-button[data-tab-group="' + group + '"]').forEach(function (btn) {
      btn.classList.remove('active');
    });
    button.classList.add('active');

    document.querySelectorAll('.tab-panel[data-tab-group="' + group + '"]').forEach(function (panel) {
      panel.classList.remove('active');
    });
    const panel = document.getElementById(target);
    if (panel) panel.classList.add('active');
  });
})();"""

ANCHOR_NAV_SCRIPT: str = """\
(function () {
  document.addEventListener('click', function (event) {
    var link = event.target.closest('a[href^="#"]');
    if (!link) return;

    var targetId = link.getAttribute('href').slice(1);
    if (!targetId) return;

    var target = document.getElementById(targetId);
    if (!target) return;

    // Walk up from target to find enclosing tab-panel
    var panel = target.closest('.tab-panel');
    if (!panel) return;
    if (panel.classList.contains('active')) return;

    var group = panel.getAttribute('data-tab-group');
    var panelId = panel.id;
    if (!group || !panelId) return;

    var button = document.querySelector(
      '.tab-button[data-tab-group="' + group + '"][data-tab-target="' + panelId + '"]'
    );
    if (!button) return;

    event.preventDefault();
    button.click();

    // Scroll after the panel becomes visible
    setTimeout(function () {
      target.scrollIntoView({behavior: 'smooth', block: 'start'});
    }, 50);
  });
})();"""

TRACE_FILTER_SCRIPT: str = """\
(function () {
  function isTraceVisible(trace) {
    return trace.visible !== false && trace.visible !== 'legendonly';
  }

  function setTraceVisible(gd, idx, visible) {
    if (!window.Plotly) return;
    window.Plotly.restyle(gd, {visible: visible ? true : 'legendonly'}, [idx]);
  }

  function groupKey(trace, idx) {
    const legendgroup = trace && trace.legendgroup != null ? String(trace.legendgroup).trim() : '';
    if (legendgroup) return 'legendgroup::' + legendgroup;
    const name = trace && trace.name != null ? String(trace.name).trim() : '';
    if (name) return 'name::' + name;
    return 'trace::' + idx;
  }

  function buildControls(controls, gd) {
    controls.innerHTML = '';
    const list = document.createElement('div');
    list.className = 'trace-filter-list';
    controls.appendChild(list);

    const groups = {};
    gd.data.forEach(function (trace, idx) {
      const key = groupKey(trace, idx);
      if (!groups[key]) {
        groups[key] = {
          indices: [],
          label: '',
          hasLegendEntry: false,
        };
      }
      groups[key].indices.push(idx);
      if (trace && trace.showlegend !== false && !groups[key].hasLegendEntry) {
        groups[key].label = trace.name || ('Trace ' + (idx + 1));
        groups[key].hasLegendEntry = true;
      }
    });

    const entries = Object.values(groups).filter(function (g) { return g.hasLegendEntry; });
    entries.sort(function (a, b) {
      return a.indices[0] - b.indices[0];
    });

    let nItems = 0;
    entries.forEach(function (group) {
      const label = document.createElement('label');
      label.className = 'trace-toggle';
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = group.indices.some(function (traceIdx) {
        return isTraceVisible(gd.data[traceIdx]);
      });
      checkbox.addEventListener('change', function () {
        group.indices.forEach(function (traceIdx) {
          setTraceVisible(gd, traceIdx, checkbox.checked);
        });
      });
      const text = document.createElement('span');
      text.textContent = group.label;

      label.appendChild(checkbox);
      label.appendChild(text);
      list.appendChild(label);
      nItems += 1;
    });

    if (nItems < 2) {
      controls.remove();
      return;
    }
  }

  function setupContainer(container) {
    if (!container || container.getAttribute('data-trace-filters-ready') === '1') return;
    const gd = container.querySelector('.plotly-graph-div');
    const controls = container.querySelector('.trace-filter-controls');
    if (!gd || !controls || !gd.data || !Array.isArray(gd.data)) return;

    const render = function () {
      buildControls(controls, gd);
    };

    if (typeof gd.on === 'function') {
      gd.on('plotly_afterplot', render);
      gd.on('plotly_restyle', render);
    }
    window.addEventListener('resize', render);

    render();
    setTimeout(render, 250);
    setTimeout(render, 1000);

    container.setAttribute('data-trace-filters-ready', '1');
  }

  function initTraceFilters() {
    document.querySelectorAll('.figure-container.has-trace-filters').forEach(setupContainer);
  }

  window.addEventListener('load', function () {
    initTraceFilters();
    setTimeout(initTraceFilters, 250);
    setTimeout(initTraceFilters, 1000);
  });
})();"""

TABLE_SORT_SCRIPT: str = """\
(function () {
  function parseComparable(text) {
    if (text == null) return {type: 'text', value: ''};
    const raw = String(text).trim();
    const cleaned = raw.replace(/,/g, '');
    const match = cleaned.match(/^-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?/);
    if (match) {
      const num = Number(match[0]);
      if (!Number.isNaN(num)) return {type: 'number', value: num};
    }
    return {type: 'text', value: raw.toLowerCase()};
  }

  function compareCells(aText, bText) {
    const a = parseComparable(aText);
    const b = parseComparable(bText);
    if (a.type === 'number' && b.type === 'number') return a.value - b.value;
    return String(a.value).localeCompare(String(b.value), undefined, {numeric: true, sensitivity: 'base'});
  }

  function setHeaderState(th, state) {
    th.setAttribute('data-sort-state', state);
    const indicator = th.querySelector('.sort-indicator');
    if (!indicator) return;
    indicator.textContent = state === 'desc' ? '\u2193' : (state === 'asc' ? '\u2191' : '\u2195');
  }

  function resetOtherHeaders(table, activeTh) {
    table.querySelectorAll('thead th.sortable-header').forEach(function (th) {
      if (th !== activeTh) setHeaderState(th, 'none');
    });
  }

  function sortTableByColumn(table, colIdx, state) {
    const tbody = table.querySelector('tbody');
    if (!tbody) return;
    const rows = Array.from(tbody.querySelectorAll('tr'));

    let sortedRows;
    if (state === 'none') {
      sortedRows = rows.slice().sort(function (a, b) {
        return Number(a.getAttribute('data-original-index')) - Number(b.getAttribute('data-original-index'));
      });
    } else {
      const sign = state === 'desc' ? -1 : 1;
      sortedRows = rows.slice().sort(function (rowA, rowB) {
        const cellA = rowA.children[colIdx];
        const cellB = rowB.children[colIdx];
        const textA = cellA ? cellA.textContent : '';
        const textB = cellB ? cellB.textContent : '';
        return sign * compareCells(textA, textB);
      });
    }

    sortedRows.forEach(function (row) { tbody.appendChild(row); });
  }

  function initTable(table) {
    if (!table || table.getAttribute('data-sortable-ready') === '1') return;
    if (table.getAttribute('data-grouped-header') === '1') return;
    const tbody = table.querySelector('tbody');
    const headerCells = table.querySelectorAll('thead th');
    if (!tbody || !headerCells.length) return;

    Array.from(tbody.querySelectorAll('tr')).forEach(function (row, idx) {
      row.setAttribute('data-original-index', String(idx));
    });

    headerCells.forEach(function (th, colIdx) {
      th.classList.add('sortable-header');
      th.setAttribute('data-sort-state', 'none');
      th.setAttribute('title', 'Sort: descending, ascending, original');

      const label = document.createElement('span');
      label.className = 'sort-label';
      label.textContent = th.textContent;

      const indicator = document.createElement('span');
      indicator.className = 'sort-indicator';
      indicator.textContent = '\u2195';

      th.textContent = '';
      th.appendChild(label);
      th.appendChild(indicator);

      th.addEventListener('click', function () {
        const current = th.getAttribute('data-sort-state') || 'none';
        const next = current === 'none' ? 'desc' : (current === 'desc' ? 'asc' : 'none');
        resetOtherHeaders(table, th);
        setHeaderState(th, next);
        sortTableByColumn(table, colIdx, next);
      });
    });

    table.setAttribute('data-sortable-ready', '1');
  }

  function initGroupedTable(table) {
    if (!table || table.getAttribute('data-sortable-ready') === '1') return;
    if (table.getAttribute('data-grouped-header') !== '1') return;
    const tbody = table.querySelector('tbody');
    const colHeaders = table.querySelectorAll('.column-header-row th');
    if (!tbody || !colHeaders.length) return;

    colHeaders.forEach(function (th) {
      var colIdx = Number(th.getAttribute('data-col-idx'));
      th.classList.add('sortable-header');
      th.setAttribute('data-sort-state', 'none');
      th.setAttribute('title', 'Sort: descending, ascending, original');

      var label = document.createElement('span');
      label.className = 'sort-label';
      label.textContent = th.textContent;

      var indicator = document.createElement('span');
      indicator.className = 'sort-indicator';
      indicator.textContent = '\u2195';

      th.textContent = '';
      th.appendChild(label);
      th.appendChild(indicator);

      th.addEventListener('click', function () {
        var current = th.getAttribute('data-sort-state') || 'none';
        var next = current === 'none' ? 'desc' : (current === 'desc' ? 'asc' : 'none');
        resetOtherHeaders(table, th);
        setHeaderState(th, next);
        sortTableByColumn(table, colIdx, next);
      });
    });

    table.setAttribute('data-sortable-ready', '1');
  }

  function initSortableTables() {
    document.querySelectorAll('.section-content table, .tab-panel table').forEach(initTable);
    document.querySelectorAll('.feature-analysis-table[data-grouped-header="1"]').forEach(initGroupedTable);
  }

  function markCompactTables() {
    document.querySelectorAll('.section-content table').forEach(function (t) {
      var ths = t.querySelectorAll('thead th');
      if (ths.length === 2) t.classList.add('compact-kv');
    });
  }

  function markTableAfterHeading(headingText, className) {
    document.querySelectorAll('.section-content h3, .section-content h4').forEach(function (h) {
      if (h.textContent.trim() !== headingText) return;
      var el = h.nextElementSibling;
      while (el && !/^H[1-4]$/.test(el.tagName)) {
        if (el.tagName === 'TABLE') {
          el.classList.add(className);
          return;
        }
        el = el.nextElementSibling;
      }
    });
  }

  window.addEventListener('load', function () {
    initSortableTables();
    markCompactTables();
    markTableAfterHeading('Feature Summary', 'feature-summary');
    markTableAfterHeading('Cohort Characteristics', 'feature-summary');
    markTableAfterHeading('Missing Data', 'compact-rows');
    markTableAfterHeading('Incremental Value', 'compact-incremental');
    markTableAfterHeading('Per-Subpopulation Incremental Value', 'compact-incremental');
    setTimeout(initSortableTables, 250);
  });
})();"""

PREDICTION_TABLE_SCROLL_SCRIPT: str = """\
(function () {
  function isVisible(element) {
    if (!element) return false;
    const rect = element.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  function applyPredictionScroll(container) {
    const table = container.querySelector('table');
    const tbody = table ? table.querySelector('tbody') : null;
    if (!table || !tbody) return;
    if (!isVisible(container)) {
      // Keep default sizing for hidden tab panels; we recompute on activation.
      container.style.maxHeight = 'none';
      container.style.overflowY = 'visible';
      return;
    }

    const rows = Array.from(tbody.querySelectorAll('tr'));
    if (!rows.length) return;

    const visibleRows = Number(container.getAttribute('data-visible-rows') || '30');
    if (rows.length <= visibleRows) {
      container.style.maxHeight = 'none';
      container.style.overflowY = 'visible';
      return;
    }

    const thead = table.querySelector('thead');
    const headH = thead ? thead.getBoundingClientRect().height : 0;
    const sampleRows = rows.slice(0, Math.min(5, rows.length));
    const avgRowH = sampleRows.reduce(function (acc, row) {
      return acc + row.getBoundingClientRect().height;
    }, 0) / sampleRows.length;
    if (!avgRowH || avgRowH <= 0) {
      container.style.maxHeight = 'none';
      container.style.overflowY = 'visible';
      return;
    }
    const maxHeight = Math.ceil(headH + avgRowH * visibleRows + 2);

    container.style.maxHeight = maxHeight + 'px';
    container.style.overflowY = 'auto';
  }

  function initPredictionTableScroll() {
    document.querySelectorAll('.prediction-table-scroll').forEach(applyPredictionScroll);
  }

  window.addEventListener('load', function () {
    initPredictionTableScroll();
    setTimeout(initPredictionTableScroll, 250);
  });
  window.addEventListener('resize', initPredictionTableScroll);
  document.addEventListener('click', function (event) {
    if (!event.target.closest('.tab-button')) return;
    // Tab script toggles classes on the same click; defer one tick and recompute.
    setTimeout(initPredictionTableScroll, 0);
  });
})();"""

FEATURE_TABLE_TOGGLE_SCRIPT: str = """\
(function () {
  function updateTableVisibility(container) {
    var table = container.querySelector('.feature-analysis-table');
    if (!table) return;

    // Collect checked states for models and types
    var modelState = {};
    container.querySelectorAll('input[data-ft-model]').forEach(function(cb) {
      modelState[cb.getAttribute('data-ft-model')] = cb.checked;
    });
    var typeState = {};
    container.querySelectorAll('input[data-ft-type]').forEach(function(cb) {
      typeState[cb.getAttribute('data-ft-type')] = cb.checked;
    });

    // Update cell visibility (column-header + tbody cells)
    table.querySelectorAll('th[data-ft-type], th[data-ft-model], td[data-ft-type], td[data-ft-model]').forEach(function(el) {
      var m = el.getAttribute('data-ft-model');
      var t = el.getAttribute('data-ft-type');
      var modelOk = !m || modelState[m] !== false;
      var typeOk = !t || typeState[t] !== false;
      el.classList.toggle('ft-hidden', !(modelOk && typeOk));
    });

    // Update group header colspans
    table.querySelectorAll('.group-header-row th[data-ft-group-id]').forEach(function(th) {
      var gid = th.getAttribute('data-ft-group-id');
      var visible = table.querySelectorAll(
        '.column-header-row th[data-ft-group-id="' + gid + '"]:not(.ft-hidden)'
      ).length;
      if (visible > 0) {
        th.setAttribute('colspan', visible);
        th.classList.remove('ft-hidden');
      } else {
        th.classList.add('ft-hidden');
      }
    });
  }

  function initFeatureTableToggles() {
    document.querySelectorAll('.feature-table-controls input[data-ft-model], .feature-table-controls input[data-ft-type]').forEach(function(cb) {
      cb.addEventListener('change', function() {
        var container = cb.closest('.feature-table-container');
        if (container) updateTableVisibility(container);
      });
    });
  }

  function initFeatureTableExpand() {
    document.querySelectorAll('.ft-expand-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        var container = btn.closest('.feature-table-container');
        if (!container) return;
        var expanded = container.classList.toggle('ft-expanded');
        var nTotal = btn.getAttribute('data-n-total');
        var nVisible = btn.getAttribute('data-n-visible');
        var collapsedText = btn.getAttribute('data-collapsed-text');
        var expandedText = btn.getAttribute('data-expanded-text');
        if (collapsedText && expandedText) {
          btn.textContent = expanded ? expandedText : collapsedText;
        } else {
          btn.textContent = expanded
            ? 'Show top ' + nVisible + ' features'
            : 'Show all ' + nTotal + ' features';
        }
      });
    });
  }

  window.addEventListener('load', function () {
    initFeatureTableToggles();
    initFeatureTableExpand();
    setTimeout(initFeatureTableToggles, 250);
  });
})();"""
