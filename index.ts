#!/usr/bin/env npx tsx
/**
 * Repro: streamText + withMastra does not persist message history or embeddings
 *
 * Expected: both generateText and streamText persist messages AND embeddings
 * Actual:   only generateText persists; streamText leaves both storage empty
 *
 * Issue: https://github.com/mastra-ai/mastra/issues/13414
 * PR fix: https://github.com/mastra-ai/mastra/pull/13297
 *
 * Run:
 *   OPENAI_API_KEY=sk-... npx tsx repro-stream-persist.ts
 *
 * Dependencies (add to a package.json or install globally):
 *   pnpm add ai @ai-sdk/openai @mastra/ai-sdk @mastra/libsql
 */

import { createOpenAI } from '@ai-sdk/openai';
import { withMastra } from '@mastra/ai-sdk';
import { LibSQLStore, LibSQLVector } from '@mastra/libsql';
import type { MemoryStorage } from '@mastra/core/storage';
import type { MastraVector } from '@mastra/core/vector';
import { generateText, streamText } from 'ai';

const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODEL = 'gpt-5-mini';
const EMBED_MODEL = 'text-embedding-3-small';
const INDEX_NAME = 'repro_embeddings'; // explicit name so we always know what to query
const RESOURCE_ID = 'repro-user';

// ── helpers ──────────────────────────────────────────────────────────────────

async function makeThread(mem: MemoryStorage, id: string) {
  await mem.saveThread({
    thread: {
      id,
      resourceId: RESOURCE_ID,
      title: id,
      metadata: {},
      createdAt: new Date(),
      updatedAt: new Date(),
    },
  });
}

async function countMessages(mem: MemoryStorage, threadId: string): Promise<number> {
  const { messages } = await mem.listMessages({ threadId });
  return messages.length;
}

async function countEmbeddings(vector: MastraVector): Promise<number> {
  try {
    const indexes = await vector.listIndexes();
    if (!indexes.includes(INDEX_NAME)) return 0;
    const { count } = await vector.describeIndex({ indexName: INDEX_NAME });
    return count;
  } catch {
    return 0;
  }
}

function row(label: string, msgs: number, embeddings: number) {
  const mOk = msgs >= 2   ? '✅' : '❌';
  const eOk = embeddings >= 1 ? '✅' : '❌';
  console.log(`  ${label.padEnd(14)} messages ${String(msgs).padStart(2)} ${mOk}   embeddings ${String(embeddings).padStart(2)} ${eOk}`);
}

// ── main ─────────────────────────────────────────────────────────────────────

async function main() {
  // Separate in-memory SQLite instances — each class manages its own connection
  const storage = new LibSQLStore({ id: 'repro', url: 'file::memory:' });
  await storage.init();
  const mem = (await storage.getStore('memory')) as MemoryStorage;

  // LibSQLVector also uses in-memory SQLite; same-process connection is stable
  const vector = new LibSQLVector({ url: 'file::memory:' });

  const base        = openai(MODEL);
  const embedder    = openai.embedding(EMBED_MODEL);
  const semanticRecall = { vector, embedder, topK: 3, messageRange: 1, indexName: INDEX_NAME };

  console.log('\nRepro: message history + embeddings after generateText vs streamText\n');
  console.log(`  ${''.padEnd(14)} messages       embeddings`);
  console.log(`  ${'─'.repeat(50)}`);

  // ── generateText ────────────────────────────────────────────────────────
  await makeThread(mem, 'thread-generate');
  const genModel = withMastra(base, {
    memory: {
      storage: mem,
      threadId: 'thread-generate',
      resourceId: RESOURCE_ID,
      lastMessages: 10,
      semanticRecall,
    },
  });

  await generateText({ model: genModel, prompt: 'Say "pong" and nothing else.' });

  const genMsgs  = await countMessages(mem, 'thread-generate');
  const genEmbed = await countEmbeddings(vector);
  row('generateText', genMsgs, genEmbed);

  // ── streamText ──────────────────────────────────────────────────────────
  await makeThread(mem, 'thread-stream');
  const streamModel = withMastra(base, {
    memory: {
      storage: mem,
      threadId: 'thread-stream',
      resourceId: RESOURCE_ID,
      lastMessages: 10,
      semanticRecall,
    },
  });

  const { textStream } = streamText({ model: streamModel, prompt: 'Say "pong" and nothing else.' });
  // Must drain the stream — flush() (which calls processOutputResult) only runs after the last chunk
  for await (const _ of textStream) { /* drain */ }

  const streamMsgs  = await countMessages(mem, 'thread-stream');
  const streamEmbed = await countEmbeddings(vector);
  row('streamText ', streamMsgs, streamEmbed - genEmbed); // delta: only embeddings added by this call

  // ── verdict ─────────────────────────────────────────────────────────────
  console.log();
  const pass = genMsgs >= 2 && genEmbed >= 1 && streamMsgs >= 2 && (streamEmbed - genEmbed) >= 1;
  if (pass) {
    console.log('  PASS — both calls persisted messages and embeddings\n');
  } else {
    console.log('  FAIL — streamText did not call processOutputResult (bug confirmed)\n');
  }
}

main().catch(err => { console.error(err); process.exit(1); });

