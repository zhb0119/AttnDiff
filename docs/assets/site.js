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

// Keep the full matrix and model directory readable without JavaScript.
const matrixButtons = [...document.querySelectorAll('[data-matrix-filter]')];
const matrixRows = [...document.querySelectorAll('[data-study]')];
function filterMatrix(group) {
  let visible = 0;
  matrixRows.forEach(row => {
    row.hidden = group !== 'all' && row.dataset.study !== group;
    if (!row.hidden) visible += 1;
  });
  matrixButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.matrixFilter === group)));
  document.querySelector('#matrix-status').textContent = `${visible} of ${matrixRows.length} experimental settings`;
}
matrixButtons.forEach(button => button.addEventListener('click', () => filterMatrix(button.dataset.matrixFilter)));
document.querySelector('.matrix-controls').hidden = false;
filterMatrix('core');

const modelCards = [...document.querySelectorAll('.model-card')];
const modelSearch = document.querySelector('#model-search');
const modelCategory = document.querySelector('#model-category');
const moreModels = document.querySelector('#more-models');
const pageSize = 9;
let visibleLimit = pageSize;
const modelText = new Map(modelCards.map(card => [card, card.textContent.toLowerCase()]));
function filterModels(reset = true) {
  if (reset) visibleLimit = pageSize;
  const terms = modelSearch.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const matches = modelCards.filter(card =>
    (modelCategory.value === 'all' || card.dataset.category === modelCategory.value) &&
    terms.every(term => modelText.get(card).includes(term))
  );
  const visible = new Set(matches.slice(0, visibleLimit));
  modelCards.forEach(card => { card.hidden = !visible.has(card); });
  document.querySelector('#model-count').textContent = `Showing ${visible.size} of ${matches.length} matching models · ${modelCards.length} entries total`;
  document.querySelector('#model-empty').hidden = matches.length > 0;
  moreModels.hidden = matches.length <= visibleLimit;
  moreModels.textContent = `Show ${Math.max(0, Math.min(pageSize, matches.length - visibleLimit))} more models ↓`;
}
modelSearch.addEventListener('input', () => filterModels());
modelCategory.addEventListener('change', () => filterModels());
document.querySelector('#reset-models').addEventListener('click', () => {
  modelSearch.value = '';
  modelCategory.value = 'all';
  filterModels();
  modelSearch.focus();
});
moreModels.addEventListener('click', () => {
  const previous = new Set(modelCards.filter(card => !card.hidden));
  visibleLimit += pageSize;
  filterModels(false);
  // Move focus to the first revealed entry, including cards without a checkpoint link.
  const revealed = modelCards.find(card => !card.hidden && !previous.has(card));
  if (revealed) {
    revealed.tabIndex = -1;
    revealed.focus({ preventScroll: true });
  }
});
document.querySelectorAll('[data-model-category]').forEach(link => {
  link.addEventListener('click', () => {
    modelCategory.value = link.dataset.modelCategory;
    modelSearch.value = '';
    filterModels();
    modelSearch.focus({ preventScroll: true });
  });
});
document.querySelector('.catalog-toolbar').hidden = false;
filterModels();
