# streaming-with-mastra-bug-repro

Reproduces a bug where `streamText` + `withMastra` does not persist message history or embeddings, while `generateText` does.

- **Issue:** https://github.com/mastra-ai/mastra/issues/13414
- **PR fix:** https://github.com/mastra-ai/mastra/pull/13297

## Setup

```sh
cp .env.example .env
# fill in OPENAI_API_KEY in .env
pnpm install
```

## Run

```sh
pnpm start
```

### Expected output when bug is present

```
  generateText   messages  2 ✅   embeddings  1 ✅
  streamText     messages  0 ❌   embeddings  0 ❌

  FAIL — streamText did not call processOutputResult (bug confirmed)
```

SQLite files (`repro-storage.db`, `repro-vector.db`) are written to the `db` directory and persist between runs. Delete them for a clean slate.
