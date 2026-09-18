// Shared left navigation sidebar for the SeaGraph AI mockup pages.
function renderSidebar(active) {
  const items = [
    { key: 'dashboard', href: 'dashboard.html', label: '대시보드', icon: '<rect x="3" y="3" width="6" height="6" rx="1.3"/><rect x="11" y="3" width="6" height="6" rx="1.3"/><rect x="3" y="11" width="6" height="6" rx="1.3"/><rect x="11" y="11" width="6" height="6" rx="1.3"/>' },
    { key: 'graph', href: 'graph-analysis.html', label: '원인 그래프 분석', icon: '<circle cx="5" cy="15" r="2.3"/><circle cx="15" cy="15" r="2.3"/><circle cx="10" cy="5" r="2.3"/><line x1="6.8" y1="13.4" x2="9" y2="7"/><line x1="13.2" y1="13.4" x2="11" y2="7"/><line x1="7.3" y1="15" x2="12.7" y2="15"/>' },
    { key: 'map', href: 'map-view.html', label: '실시간 항적 지도', icon: '<path d="M10 18s6-5.8 6-10.3A6 6 0 0 0 4 7.7C4 12.2 10 18 10 18z"/><circle cx="10" cy="7.6" r="2"/>' },
  ];

  const rows = items
    .map((item) => {
      const activeClass = item.key === active ? ' is-active' : '';
      return `<a href="${item.href}">
        <div class="nav-item${activeClass}">
          <span class="nav-icon"><svg width="19" height="19" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${item.icon}</svg></span>
          <span class="nav-label">${item.label}</span>
        </div>
      </a>`;
    })
    .join('');

  return `
    <div class="sidebar">
      <div class="sidebar-brand">
        <div class="brand-icon">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="var(--color-white)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 12.5c1.6 1.3 3.2 1.3 4.8 0 1.6-1.3 3.2-1.3 4.8 0 1.6 1.3 3.2 1.3 4.8 0"/>
            <path d="M4.5 12.5 5.3 6.8h9.4l0.8 5.7"/>
            <path d="M10 6.8V3.2h2.6"/>
          </svg>
        </div>
        <div>
          <div class="brand-name">SeaGraph AI</div>
          <div class="brand-tagline">선박 지연 원인 추적 시스템</div>
        </div>
      </div>
      <div class="sidebar-divider"></div>
      <div class="nav-list">${rows}</div>
    </div>
  `;
}

function mountSidebar(active) {
  const root = document.getElementById('sidebar-root');
  if (root) root.innerHTML = renderSidebar(active);
}
