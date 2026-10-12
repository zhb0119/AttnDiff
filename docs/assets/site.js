const tabs = [...document.querySelectorAll('[role="tab"]')];
const panel = document.querySelector('#experiment-panel');
function showExperiment(tab, focus = false) {
  const data = window.ATTNDIFF_RESULTS[tab.dataset.experiment];
  tabs.forEach(item => {
    const active = item === tab;
    item.setAttribute('aria-selected', String(active));
    item.tabIndex = active ? 0 : -1;
  });
  panel.setAttribute('aria-labelledby', tab.id);
  document.querySelector('#experiment-title').textContent = data.title;
  document.querySelector('#experiment-description').textContent = data.description;
  document.querySelector('#experiment-source').textContent = data.source;
  document.querySelector('#experiment-takeaway').textContent = data.takeaway;
  const body = document.querySelector('#experiment-rows');
  body.replaceChildren(...data.rows.map(([name, ours, reef]) => {
    const row = document.createElement('tr');
    const label = document.createElement('th');
    label.scope = 'row'; label.textContent = name; row.append(label);
    [ours, reef].forEach((value, index) => {
      const cell = document.createElement('td');
      const score = document.createElement('span');
      score.className = index === 0 ? 'score ours' : 'score';
      score.style.setProperty('--score', `${value * 100}%`);
      score.textContent = value.toFixed(4); cell.append(score); row.append(cell);
    });
    return row;
  }));
  if (focus) tab.focus();
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => showExperiment(tab));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) { event.preventDefault(); showExperiment(tabs[next], true); }
  });
});
showExperiment(tabs[0]);
document.querySelector('#copy-citation').addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  const citation = document.querySelector('#bibtex');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    status.textContent = 'BibTeX copied.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange(); range.selectNodeContents(citation);
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or Command+C to copy.';
  }
});
