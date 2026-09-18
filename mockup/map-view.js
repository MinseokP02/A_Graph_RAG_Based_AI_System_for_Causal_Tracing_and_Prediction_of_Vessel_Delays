const REGION_LABELS = { all: '표시 중 128척 (전체)', apac: '표시 중 84척 (아시아-태평양)', eu: '표시 중 26척 (유럽)', na: '표시 중 18척 (북미)' };

function setRegion(region) {
  document.getElementById('region-count-label').textContent = REGION_LABELS[region];
  ['all', 'apac', 'eu', 'na'].forEach((r) => {
    document.getElementById('pill-' + r).classList.toggle('is-active', r === region);
  });
}

mountSidebar('map');
setRegion('all');
