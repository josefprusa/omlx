# Raw Source Policy

Raw sources are optional supporting material for wiki pages. Keep them small, curated, and safe to commit.

## Allowed

- Public or internal project notes that have been reviewed for safe repository storage.
- Short excerpts that explain a durable project decision, behavior, or domain rule.
- De-secreted source snippets with clear provenance and a linked wiki page.

## Denied

Do not store secrets, credentials, auth tokens, private customer data, full logs, database exports, generated dumps, or other unsafe material in `.llm-wiki/raw/`.

## Before Adding Raw Material

- Confirm the material is intentionally selected and safe to commit.
- Redact credentials, customer data, and operationally sensitive details.
- Prefer a concise summary in a wiki page when raw material is not necessary.
- Link curated raw material from the page that uses it.
