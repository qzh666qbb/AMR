# TMLR Submission Draft

Status: anonymous unified draft, editorially revised on 2026-10-25 using
unchanged frozen evidence; not externally submitted.

The manuscript combines the SWAN segmentation-interface audit and the
SemStamp/SimMark exact-resegmentation framework in one submission. The
evidence streams share a research question but do not substitute for one
another's system-specific endpoints. In particular, the draft reports no
final real certified rate.

## Editorial Revision

The 2026-10-25 revision attributes the known segmentation risk, original
z-statistic, elementary unit-count ceiling, and existing alignment DP to
their sources. It adds a SWAN--AliMark--exact-evaluation comparison of search
objects, DP states, objective functions, and guarantee scopes. Contributions
are limited to the bounded case study, declared exact-evaluation contract,
and audited implementation record.

See [EDITORIAL_REVISION.md](EDITORIAL_REVISION.md) for the concrete changes.
The previous manuscript, PDF, build record, bibliography, and README are
preserved under
`report_assets/manuscript_revision_20261025/before/` at the workspace root.
Experimental claims, endpoints, and historical submission decisions are
unchanged.

## Build

From this directory:

```bash
tectonic main.tex
```

Expected output: `main.pdf`.

For an isolated build, copy `main.tex`, `references.bib`, `tmlr.sty`,
`tmlr.bst`, and `fancyhdr.sty` to a temporary directory, compile there, and
copy only the resulting PDF back. The current `BUILD.json` records the
verified isolated build.

The parent `MANIFEST.sha256` is a historical evidence snapshot, including
the pre-revision manuscript hashes. It is not a manifest for the revised
PDF. Current editorial files are indexed by `EDITORIAL_MANIFEST.sha256`;
the revision audit and previous-file mapping are under
`report_assets/manuscript_revision_20261025/`.

## Template Provenance

- Official source:
  `https://github.com/JmlrOrg/tmlr-style-file`
- Frozen HEAD:
  `7bf90efe3a0debbba703c05c43f3ff7e4d4a2992`
- Downloaded archive SHA256:
  `48f9f972cc812d2cfc8032a78578c2f5d8939b92fbd4f61095e8cbeb3394f92d`
- Included upstream files:
  `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, `LICENSE`

The upstream style files are unmodified.

## Submission Boundary

Before external submission:

1. confirm authorship, affiliations, conflicts, funding and ethics fields;
2. review every sentence and citation manually;
3. prepare an anonymous redistributable artifact without sealed text,
   credentials or restricted third-party payloads;
4. replace the placeholder OpenReview URL only after a submission exists.

No new experiment is authorized by this draft.
