const config = window.TEDI_CONFIG;
const paperLink = document.getElementById('paper-link');
if (config.arxivUrl) {
  paperLink.href = config.arxivUrl;
  paperLink.target = '_blank';
  paperLink.rel = 'noopener';
  paperLink.removeAttribute('aria-disabled');
  paperLink.removeAttribute('title');
}
for (const year of ['2017', '2018']) {
  const video = document.getElementById('video-' + year);
  const placeholder = document.getElementById('placeholder-' + year);
  video.addEventListener('loadedmetadata', () => {
    video.hidden = false;
    placeholder.hidden = true;
  });
  video.addEventListener('error', () => {
    video.hidden = true;
    placeholder.hidden = false;
  });
  video.src = config.videos[year];
}
const copyButton = document.getElementById('copy-citation');
copyButton.addEventListener('click', async () => {
  const text = document.getElementById('bibtex').textContent;
  try {
    await navigator.clipboard.writeText(text);
    copyButton.textContent = 'Copied';
    document.getElementById('copy-status').textContent = 'BibTeX copied to clipboard.';
    setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 2000);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('bibtex'));
    const selection = window.getSelection();
    selection.removeAllRanges(); selection.addRange(range);
    document.getElementById('copy-status').textContent = 'Citation selected. Please copy the selected text.';
    copyButton.textContent = 'Select and copy';
  }
});
