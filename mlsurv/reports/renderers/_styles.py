"""CSS and SVG constants for the HTML report renderer."""

REPORT_CSS: str = """\
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --background-color: #f8f9fa;
            --card-background: #ffffff;
            --text-color: #333333;
            --border-color: #e0e0e0;
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--background-color);
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .report-header {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            gap: 10px;
            border-bottom: 3px solid var(--secondary-color);
            padding-bottom: 15px;
            margin-bottom: 30px;
        }

        .report-logo {
            flex-shrink: 0;
            width: 60px;
            height: 60px;
        }

        .report-title {
            color: var(--primary-color);
            margin: 0;
            text-align: center;
        }

        .report-metadata {
            color: #6c757d;
            font-size: 0.85em;
            margin-top: 4px;
            text-align: center;
        }

        .section-content a {
            color: var(--secondary-color);
            text-decoration: none;
        }

        .section-content a:hover {
            text-decoration: underline;
        }

        .report-section {
            background-color: var(--card-background);
            padding: 25px 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 25px;
        }

        .report-section h2 {
            color: var(--primary-color);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
            margin-top: 0;
            text-align: center;
        }

        .section-content {
            margin-bottom: 20px;
        }

        .section-content h3,
        .section-content h4,
        .section-content h5,
        .tab-panel h3,
        .tab-panel h4,
        .tab-panel h5 {
            text-align: center;
        }

        .section-content h3,
        .tab-panel h3 {
            font-size: 1.45em;
            margin-top: 40px;
            margin-bottom: 16px;
        }

        .section-content h4,
        .tab-panel h4 {
            font-size: 1.25em;
            margin-top: 35px;
            margin-bottom: 14px;
        }

        .section-content h5,
        .tab-panel h5 {
            font-size: 1.1em;
            margin-top: 28px;
            margin-bottom: 12px;
        }

        .figure-container {
            margin: 20px 0;
            padding: 10px;
            background-color: var(--background-color);
            border-radius: 4px;
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
        }

        .trace-filter-controls {
            width: 100%;
            margin-bottom: 8px;
            padding: 6px 8px;
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid var(--border-color);
            border-radius: 5px;
        }

        .trace-filter-list {
            display: flex;
            flex-wrap: wrap;
            gap: 4px 10px;
            justify-content: center;
        }

        .trace-toggle {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            font-size: 0.82em;
            color: var(--text-color);
            cursor: pointer;
        }

        .figure-container .plotly-graph-div {
            margin-left: auto !important;
            margin-right: auto !important;
        }

        .section-content .figure-placeholder {
            display: flex;
            justify-content: center;
        }

        .phase-tabs {
            margin-top: 15px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            background: #fff;
        }

        .report-sections-tabs {
            margin-top: 0;
            margin-bottom: 25px;
        }

        .report-tab-buttons {
            position: sticky;
            top: 0;
            z-index: 5;
        }

        .phase-tab-buttons {
            display: flex;
            flex-wrap: wrap;
            border-bottom: 1px solid var(--border-color);
            background: #f5f7fa;
        }

        .tab-button {
            border: none;
            background: transparent;
            padding: 10px 16px;
            cursor: pointer;
            font-weight: 600;
            color: var(--primary-color);
        }

        .tab-button.active {
            background: #fff;
            color: var(--secondary-color);
            border-bottom: 2px solid var(--secondary-color);
        }

        .tab-panel {
            display: none;
            padding: 14px;
        }

        .tab-panel.active {
            display: block;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.95em;
        }

        .prediction-table-scroll {
            width: 100%;
            overflow-x: auto;
            border: 1px solid var(--border-color);
            border-radius: 5px;
            background: #fff;
            margin-top: 8px;
        }

        .prediction-table-scroll table {
            margin: 0;
            width: 100%;
        }

        /* Feature analysis interactive table */
        .feature-table-container {
            margin: 15px 0;
        }

        .feature-table-controls {
            display: flex;
            flex-wrap: wrap;
            gap: 6px 16px;
            padding: 8px 12px;
            margin-bottom: 8px;
            background: #f5f7fa;
            border: 1px solid var(--border-color);
            border-radius: 5px;
            justify-content: center;
        }

        .feature-group-toggle {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            font-size: 0.85em;
            cursor: pointer;
            white-space: nowrap;
            user-select: none;
        }

        .ft-control-label {
            font-size: 0.85em;
            font-weight: 600;
            color: var(--secondary-color);
            white-space: nowrap;
        }

        .ft-control-separator {
            width: 1px;
            height: 18px;
            background: var(--border-color);
            margin: 0 4px;
        }

        .mlsurv-callout-warning {
            border-left: 4px solid #e6a817;
            background-color: #fef9e7;
            padding: 12px 16px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
            font-size: 0.95em;
            line-height: 1.5;
        }

        .mlsurv-metric-caveat {
            color: #a06000;
            font-size: 0.9em;
            font-style: italic;
            margin: 8px 0 15px 0;
        }

        .mlsurv-callout-info {
            border-left: 4px solid var(--secondary-color);
            background-color: #ebf5fb;
            padding: 12px 16px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
            font-size: 0.95em;
            line-height: 1.5;
        }
        .mlsurv-callout-info ul {
            margin: 5px 0 0 0;
            padding-left: 20px;
        }
        .mlsurv-callout-info li {
            margin-bottom: 4px;
        }

        .feature-table-scroll {
            width: 100%;
            overflow-x: auto;
            border: 1px solid var(--border-color);
            border-radius: 5px;
        }

        .feature-analysis-table {
            width: max-content;
            min-width: 100%;
            margin: 0;
            border-collapse: collapse;
        }

        .feature-analysis-table .group-header-row th {
            text-align: center;
            background-color: #1a2a3a;
            color: white;
            font-weight: 700;
            border-right: 2px solid rgba(255,255,255,0.3);
            padding: 8px 12px;
        }

        .feature-analysis-table .column-header-row th {
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
            padding: 8px 12px;
            white-space: nowrap;
        }

        .feature-analysis-table .sticky-col {
            position: sticky;
            left: 0;
            z-index: 2;
            background-color: inherit;
        }

        .feature-analysis-table thead .sticky-col {
            z-index: 3;
            background-color: #1a2a3a;
        }

        .feature-analysis-table tbody tr:nth-child(even) .sticky-col {
            background-color: var(--background-color);
        }

        .feature-analysis-table tbody tr:nth-child(odd) .sticky-col {
            background-color: #fff;
        }

        .feature-analysis-table tbody tr:hover .sticky-col {
            background-color: #e8f4f8;
        }

        .feature-analysis-table .ft-hidden {
            display: none;
        }

        .ft-overflow-row {
            display: none;
        }

        .feature-table-container.ft-expanded .ft-overflow-row {
            display: table-row;
        }

        .feature-table-expand {
            text-align: center;
            margin-top: 8px;
        }

        .ft-expand-btn {
            background: #f5f7fa;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 6px 16px;
            cursor: pointer;
            font-size: 0.85em;
            color: var(--secondary-color);
        }

        .ft-expand-btn:hover {
            background: #e8f0f8;
        }

        .feature-analysis-table.compact {
            width: auto;
            min-width: 0;
            margin: 0 auto;
        }

        .feature-analysis-table.compact th,
        .feature-analysis-table.compact td {
            padding: 4px 12px;
            font-size: 0.9em;
        }

        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        th {
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
        }

        th.sortable-header {
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
        }

        th.sortable-header:hover {
            background-color: #1f3346;
        }

        th .sort-label {
            display: inline-block;
            margin-right: 6px;
        }

        th .sort-indicator {
            display: inline-block;
            width: 14px;
            text-align: center;
            opacity: 0.9;
        }

        tr:nth-child(even) {
            background-color: var(--background-color);
        }

        tr:hover {
            background-color: #e8f4f8;
        }

        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
        }

        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }

        pre code {
            padding: 0;
            background-color: transparent;
        }

        .report-footer {
            text-align: center;
            color: #666;
            padding: 20px;
            margin-top: 30px;
        }

        .report-footer hr {
            border: none;
            border-top: 1px solid var(--border-color);
            margin-bottom: 15px;
        }

        /* Consolidated section-level programmatic access dropdown */
        details.section-access {
            margin: 20px 0;
            padding: 10px 15px;
            background-color: #f8f9fa;
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }

        details.section-access summary {
            cursor: pointer;
            font-weight: 500;
            color: var(--secondary-color);
            user-select: none;
        }

        details.section-access summary:hover {
            color: var(--primary-color);
        }

        details.section-access[open] summary {
            margin-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }

        details.section-access .access-group h5 {
            margin: 15px 0 8px 0;
            color: var(--primary-color);
            font-size: 0.95em;
            text-align: left;
        }

        details.section-access .access-item {
            margin: 8px 0;
            padding-left: 10px;
            border-left: 2px solid var(--border-color);
        }

        details.section-access .access-item h6 {
            margin: 4px 0;
            font-size: 0.85em;
            color: var(--text-color);
            text-align: left;
        }

        details.section-access .access-item pre {
            margin: 2px 0;
            padding: 4px 8px;
            font-size: 0.82em;
        }

        /* Legacy per-figure access sections (kept for backward compat) */
        details.figure-access {
            margin: 15px 0;
            padding: 10px 15px;
            background-color: #f8f9fa;
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }

        details.figure-access summary {
            cursor: pointer;
            font-weight: 500;
            color: var(--secondary-color);
            user-select: none;
        }

        details.figure-access summary:hover {
            color: var(--primary-color);
        }

        details.figure-access[open] summary {
            margin-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }

        details.figure-access .code-section {
            margin-top: 10px;
        }

        details.figure-access .code-section h5 {
            margin: 15px 0 8px 0;
            color: var(--primary-color);
            font-size: 0.9em;
        }

        details.figure-access .code-section h5:first-child {
            margin-top: 0;
        }

        details.figure-access .data-description {
            font-style: italic;
            color: #666;
            margin-bottom: 8px;
            font-size: 0.9em;
        }

        /* Collapsible environment config sections */
        details.environment-info {
            margin: 15px 0;
            padding: 10px 15px;
            background-color: #f8f9fa;
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }

        details.environment-info summary {
            cursor: pointer;
            font-weight: 500;
            color: var(--secondary-color);
            user-select: none;
        }

        details.environment-info summary:hover {
            color: var(--primary-color);
        }

        details.environment-info[open] summary {
            margin-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }

        /* Compact key-value tables (2-column summary tables) */
        table.compact-kv {
            width: auto;
            min-width: 340px;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        }

        table.compact-kv td,
        table.compact-kv th {
            padding: 4px 12px;
        }

        /* Compact incremental value table */
        table.compact-incremental {
            width: auto;
            max-width: 1100px;
            margin: 8px auto;
            font-size: 0.88em;
        }

        table.compact-incremental td,
        table.compact-incremental th {
            padding: 4px 12px;
        }

        /* Compact PH test table */
        .ph-test-table {
            width: auto;
            max-width: 520px;
            margin: 8px auto;
            font-size: 0.88em;
            border-collapse: collapse;
        }

        .ph-test-table th,
        .ph-test-table td {
            padding: 3px 10px;
            border-bottom: 1px solid var(--border-color);
        }

        .ph-test-table thead th {
            font-weight: 600;
            border-bottom: 2px solid var(--border-color);
        }

        .ph-test-table .ph-sig {
            color: #c0392b;
            font-weight: 600;
        }

        /* Hyperparameter selection table */
        .hp-params-table {
            width: auto;
            max-width: 900px;
            margin: 8px auto;
            font-size: 0.88em;
            border-collapse: collapse;
        }

        .hp-params-table th,
        .hp-params-table td {
            padding: 4px 10px;
            border-bottom: 1px solid var(--border-color);
        }

        .hp-params-table thead th {
            font-weight: 600;
            border-bottom: 2px solid var(--border-color);
        }

        /* Feature summary table — compact and centered */
        table.feature-summary {
            width: auto;
            max-width: 85%;
            margin-left: auto;
            margin-right: auto;
        }

        table.feature-summary td,
        table.feature-summary th {
            padding: 4px 12px;
        }

        /* Compact row tables (missing data, etc.) */
        table.compact-rows {
            width: auto;
            max-width: 70%;
            margin-left: auto;
            margin-right: auto;
        }

        table.compact-rows td,
        table.compact-rows th {
            padding: 4px 12px;
        }

        /* Key findings highlight box (blockquote) */
        .section-content blockquote {
            background-color: #ebf5fb;
            border-left: 4px solid var(--secondary-color);
            border-radius: 4px;
            padding: 15px 20px;
            margin: 15px 0;
        }

        .section-content blockquote p:first-child {
            margin-top: 0;
            font-weight: 600;
            color: var(--primary-color);
        }

        .section-content blockquote ul {
            margin: 0;
            padding-left: 20px;
        }

        .section-content blockquote li {
            margin-bottom: 4px;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }

            .report-section {
                padding: 15px;
            }

            .phase-tab-buttons {
                flex-direction: column;
            }

            .tab-button {
                width: 100%;
                text-align: left;
            }

            .trace-filter-controls {
                width: 100%;
                margin-bottom: 10px;
            }

            table {
                font-size: 0.85em;
            }

            th, td {
                padding: 8px 10px;
            }
        }"""

LOGO_SVG: str = '''\
<svg class="report-logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#023E8A"/>
      <stop offset="100%" style="stop-color:#0077B6"/>
    </linearGradient>
  </defs>
  <rect x="5" y="5" width="90" height="90" rx="18" ry="18" fill="url(#bgGrad)"/>
  <path d="M 15,20 L 28,20 L 28,28 L 52,28 L 52,36 L 64,36 L 64,44 L 76,44 L 76,50 L 85,50"
        fill="none" stroke="#ffffff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>
  <circle cx="28" cy="20" r="3" fill="#48CAE4"/>
  <circle cx="52" cy="28" r="3" fill="#48CAE4"/>
  <circle cx="64" cy="36" r="3" fill="#90E0EF"/>
  <circle cx="76" cy="44" r="3" fill="#90E0EF"/>
  <g opacity="0.85">
    <line x1="36" y1="28" x2="27" y2="39" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>
    <line x1="36" y1="28" x2="45" y2="39" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>
    <line x1="27" y1="39" x2="21" y2="50" stroke="#ffffff" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="27" y1="39" x2="33" y2="50" stroke="#ffffff" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="45" y1="39" x2="39" y2="50" stroke="#ffffff" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="45" y1="39" x2="51" y2="50" stroke="#ffffff" stroke-width="1.5" stroke-linecap="round"/>
    <circle cx="36" cy="28" r="3" fill="#ffffff"/>
    <circle cx="27" cy="39" r="2.5" fill="#ffffff"/>
    <circle cx="45" cy="39" r="2.5" fill="#ffffff"/>
    <circle cx="21" cy="50" r="2" fill="#ffffff"/>
    <circle cx="33" cy="50" r="2" fill="#ffffff"/>
    <circle cx="39" cy="50" r="2" fill="#ffffff"/>
    <circle cx="51" cy="50" r="2" fill="#ffffff"/>
  </g>
  <text x="50" y="75" font-family="'Segoe UI', Arial, sans-serif" font-size="18" font-weight="700"
        fill="#ffffff" text-anchor="middle">mlsurv</text>
</svg>'''
