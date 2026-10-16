# Project page

The static site in this directory serves the existing **https://zhb0119.github.io/AttnDiff/** project page. It preserves the original site's model search, experiment filters, result tabs, and citation copying.

## Preview locally

From the repository root:

```bash
python -m http.server 8000 --directory docs
```

Open http://localhost:8000. Use an HTTP server to test subtitle loading; local `file://` pages may block caption requests.

## Publish

The [Pages workflow](../.github/workflows/pages.yml) uploads this directory when changes under `docs/` are pushed to `main`. The repository's **Settings → Pages → Build and deployment → Source** should be **GitHub Actions**. A manual run is also available through **Actions → Deploy project page → Run workflow**.

No Node dependencies or build step are needed. All media and site assets are served locally by GitHub Pages.

## Content and media

- `index.html`: authors, affiliations, paper links, figures, demo, experiment matrix, model directory, results, and citation.
- `assets/style.css`: existing responsive design.
- `assets/presentation.css`: page layout and media refinements.
- `assets/site.js`: result tabs, search/filter controls, and citation copy.
- `assets/results.js`: paper-reported fine-tuning, merging, and pruning values.
- `assets/teaser.png`, `assets/pipeline.png`: unmodified paper artwork.
- Matching `.webp` files: optimized display derivatives; clicking a figure opens the original PNG.
- `assets/video/AttnDiff_story_60s_captioned.mp4`: supplied 60-second narrated demo with embedded English captions, copied without re-encoding. Extra subtitle tracks are off by default to avoid duplicate English captions.
- `assets/video/preview.gif`: animated preview sampled from the 60-second video, used by the README.
- `assets/video/poster.webp`: optimized poster extracted from the 60-second video.
- `assets/video/captions.en.vtt`: captions converted from the supplied 60-second narration subtitles.
- `assets/video/captions.zh.vtt`: Chinese translation of that narration.

The teaser and video depict an illustrative model-provenance scenario. The displayed families, fingerprint patterns, and highlighted match are schematic, not experimental attribution results.

Author information and new adaptive-suppression results follow the supplied LaTeX manuscript. The existing public arXiv identifier is retained. When the paper changes, update both the README and page, and keep `CITATION.cff` consistent with the BibTeX.
