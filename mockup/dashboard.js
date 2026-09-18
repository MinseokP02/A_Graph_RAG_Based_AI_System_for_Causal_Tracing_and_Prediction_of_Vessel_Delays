const RANGE_DATA = {
  '7d':  { ships:'124척', shipsDelta:'▲ 2척 전주 대비',  delay:'19.8%', delayDelta:'▲ 1.2%p 전주 대비',  hours:'11.6시간', hoursDelta:'▼ 0.9시간 개선', acc:'92.1%', accDelta:'▲ 0.6%p 모델 개선', label:'최근 7일 기준' },
  '30d': { ships:'128척', shipsDelta:'▲ 6척 전월 대비',  delay:'22.4%', delayDelta:'▲ 3.1%p 전월 대비',  hours:'14.2시간', hoursDelta:'▼ 1.8시간 개선', acc:'91.3%', accDelta:'▲ 2.4%p 모델 개선', label:'최근 30일 기준' },
  '90d': { ships:'135척', shipsDelta:'▲ 11척 전분기 대비', delay:'24.7%', delayDelta:'▲ 4.0%p 전분기 대비', hours:'16.5시간', hoursDelta:'▼ 2.6시간 개선', acc:'89.8%', accDelta:'▲ 3.1%p 모델 개선', label:'최근 90일 기준' },
};

function setRange(range) {
  const d = RANGE_DATA[range];
  document.getElementById('range-label').textContent = d.label;
  document.getElementById('kpi-ships').textContent = d.ships;
  document.getElementById('kpi-ships-delta').textContent = d.shipsDelta;
  document.getElementById('kpi-delay').textContent = d.delay;
  document.getElementById('kpi-delay-delta').textContent = d.delayDelta;
  document.getElementById('kpi-hours').textContent = d.hours;
  document.getElementById('kpi-hours-delta').textContent = d.hoursDelta;
  document.getElementById('kpi-acc').textContent = d.acc;
  document.getElementById('kpi-acc-delta').textContent = d.accDelta;
  ['7d', '30d', '90d'].forEach((r) => {
    document.getElementById('pill-' + r).classList.toggle('is-active', r === range);
  });
}

mountSidebar('dashboard');
setRange('30d');
