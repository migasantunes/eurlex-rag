# M1 checklist: what to verify against real EUR-Lex files

The fetcher and the parser were written without access to EUR-Lex. They are
tested on synthetic pages that follow the documented layout. The first job of
milestone M1 is to run `eurlex ingest` for real and work through this list.
`eurlex ingest` reports every problem it finds in one run; the alignment and
count checks are what tell you the parser is right.

## Downloading

1. Does the Cellar address (`publications.europa.eu/resource/celex/<CELEX>`
   with `Accept` and `Accept-Language` headers) return the page directly? It
   may answer "300 Multiple Choices" or return a zip. Either currently falls
   through to the EUR-Lex page address.
2. Does the EUR-Lex page address work from a script, or does it block
   automated requests? If both routes fail, download the 30 files in a browser
   into `data/raw/` as `<CELEX>.<lang>.html` and run `eurlex ingest --no-fetch`.
3. Files are read as UTF-8. Confirm no page comes in another encoding.
4. Confirm the reuse and attribution wording on EUR-Lex and put it in the
   README.

## Page layout

5. Does the 2016 GDPR page carry the structural ids (`art_6`, `rct_47`) or only
   the old class names? The parser handles both; this decides which path runs.
6. Paragraph containers: are they always `div` ids of the form `006.002`? Can
   the `p` tags carry more than one class? Only the first class is read.
7. Annexes: what id does a single unnumbered annex get? Are the "ANNEX III"
   label and the annex title both `doc-ti` lines?
8. Signature block ("Done at Brussels..."): the parser assumes classes `final`
   or `signatory`. If they differ, the closing text is glued onto the last
   article in the old-style path.
9. Footnotes: assumed `note-tag` for the marker and `p.note` for the note.

## Old-style pages only (no ids)

10. Recitals are assumed to be one table row each. Continuation lines would be
    dropped.
11. A section heading replaces the chapter heading instead of nesting under it.
12. French "Article premier" has no digit; the parser counts it as 1.
13. Only plain lines numbered in sequence become paragraph units. Numbered
    definitions ("(1) 'personal data' means...") stay inside the article.

## Counts

14. Expected articles / recitals, to confirm against the published texts:
    GDPR 99 / 173, DMA 54 / 109, DSA 93 / 156, Data Act 50 / 119,
    AI Act 113 / 180. They live in `src/eurlex_rag/ingest/checks.py`.

## When it passes

- Trim two or three real pages and add them under `tests/fixtures/` as
  regression fixtures, so the parser is tested against the real layout from
  then on.
- Open ten random articles in English, Portuguese and French side by side and
  confirm they are the same article (PLAN, M1, "Your part").
