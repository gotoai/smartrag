"""Embeddings: a provider catalog, one API, and dialects as data.

| Module | Role |
|---|---|
| `api` | `Embedder`, `open_embedder`: all the rest of Smart RAG uses |
| `catalog`, `providers.yaml` | The providers, their models and dialects |
| `resolve` | A kb's settings + the catalog → `EmbeddingSpec`, with its `VectorSpace` |
| `marking` | Query versus document: `none`, `field`, `prefix` |
| `wire` | Request and response formats: `openai` |
| `transport` | HTTP, retries, `Retry-After`, error kinds |
| `engine` | The generic `HttpEmbedder` |

Import from the modules: this package's `__init__` stays empty so that `kb.config` can
import `resolve` while `engine` imports `kb.config`.
"""
