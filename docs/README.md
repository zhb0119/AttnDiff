# AttnDiff project website

A dependency-free, responsive static site for the paper. The intended public URL is
https://zhb0119.github.io/AttnDiff/. A separate `zhb0119.github.io` repository is **not required**.

## Preview locally

From the repository root:

```bash
python -m http.server 8000 --directory docs
```

Open http://localhost:8000. You can also open `docs/index.html` directly.

## Publish with GitHub Pages

1. Commit and push `docs/`, `.github/workflows/pages.yml`, and the README changes to `main`.
2. In [repository Settings → Pages](https://github.com/zhb0119/AttnDiff/settings/pages), set **Build and deployment → Source → GitHub Actions**.
3. In **Actions → Deploy project website**, choose **Run workflow** (or push another website change).
4. Wait for the deployment to finish, then open https://zhb0119.github.io/AttnDiff/.

The workflow publishes only `docs/`. No Node installation or build step is needed.

## Content and sources

- News: September 2026 NeurIPS acceptance, supplied by the repository owner.
- Paper and author list: the existing README citation, arXiv:2604.05502.
- Experimental text and numbers: the supplied `neurips_2026.pdf`, Tables 1–4,
  9–11, and 14; Section 5.1, Section 5.6, and the computational-cost discussion.
- `assets/results.js`: transcribed values with their table references. These are
  paper-reported results, not newly executed experiments. Group averages in
  Table 11 and the 69-suspect benchmark in Table 4 are different evaluation pools.
- `assets/pipeline.webp`: optimized copy of the repository's `figure/pipeline.png`.
- `assets/probe-budget.webp` and `assets/layer-similarity.webp`: figures cropped
  from the supplied manuscript (Figures 5 and 6), without page furniture.
- The Paper PDF button links to the public arXiv PDF; the local anonymous reviewer
  manuscript is used as a content reference and is not included in the website.

Edit text in `index.html`, design in `assets/style.css`, and interactive behavior
in `assets/site.js`. Update `assets/results.js` if paper results change. All assets
are local; there are no analytics, font services, or third-party JavaScript dependencies.
