function setMode(mode) {
  document.getElementById('answer-local').classList.toggle('is-hidden', mode !== 'local');
  document.getElementById('answer-global').classList.toggle('is-hidden', mode !== 'global');
  document.getElementById('mode-label').textContent = mode === 'local' ? 'Local Search 모드' : 'Global Search 모드';
  document.getElementById('pill-local').classList.toggle('is-active', mode === 'local');
  document.getElementById('pill-global').classList.toggle('is-active', mode === 'global');
}

mountSidebar('graph');
setMode('local');
