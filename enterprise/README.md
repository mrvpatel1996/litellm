# LiteLLM Enterprise Fork

> Production-ready LiteLLM proxy with **self-hosted Langfuse observability**, **Qdrant semantic caching**, and **9-service Docker Compose stack** — one command to run everything.

## 🚀 Quick Start

```bash
# 1. Clone
git clone -b enterprise-features https://github.com/mrvpatel1996/litellm.git
cd litellm

# 2. Create .env with your API keys
cat > .env << 'EOF'
LITELLM_MASTER_KEY="sk-your-master-key"
OPENROUTER_API_KEY="sk-or-your-key"
UI_USERNAME="admin"
UI_PASSWORD="your-secure-password"
EOF

# 3. Start everything
docker compose -f docker-compose.enterprise.yml up -d

# 4. Wait ~90s, then open:
#    LiteLLM UI:  http://localhost:4000/ui
#    Langfuse UI:  http://localhost:3001
```

## 📦 Docker Image

Pre-built image available on GitHub Container Registry:

```bash
docker pull ghcr.io/mrvpatel1996/litellm-enterprise:latest
```

Multi-arch supported (amd64 + arm64).

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your Applications                     │
│              (OpenAI-compatible API calls)               │
└─────────────────────┬───────────────────────────────────┘
                      │ :4000
┌─────────────────────▼───────────────────────────────────┐
│               LiteLLM Enterprise Proxy                   │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │  Routing   │  │  Auth/Keys   │  │  Semantic Cache │ │
│  └────────────┘  └──────────────┘  └────────┬────────┘ │
│  ┌────────────┐  ┌──────────────┐           │          │
│  │  Rate Ltd  │  │  Budget Mgmt │  ┌────────▼────────┐ │
│  └────────────┘  └──────────────┘  │    Qdrant       │ │
│  ┌────────────┐                     │  Vector Store   │ │
│  │  Logging   │───► Langfuse ◄─────└─────────────────┘ │
│  └────────────┘     (all LLM calls traced)              │
└─────────────────────────────────────────────────────────┘
```

## 🧩 Services (9 containers)

| Service | Port | Purpose |
|---|---|---|
| **litellm** | `4000` | LLM proxy + Admin UI |
| **db** | `5435` | PostgreSQL for LiteLLM |
| **qdrant** | `6335` | Vector store + semantic cache |
| **langfuse-web** | `3001` | Langfuse observability UI |
| **langfuse-worker** | — | Async event processing |
| **langfuse-db** | — | PostgreSQL for Langfuse |
| **clickhouse** | — | OLAP analytics engine |
| **redis** | — | Cache + job queue |
| **minio** | `9090` | S3-compatible storage |

## 🔧 Configuration

### Environment Variables (.env)

```bash
# ── Required ──
LITELLM_MASTER_KEY="sk-your-master-key"       # API gateway auth
OPENROUTER_API_KEY="sk-or-your-key"            # LLM provider key

# ── LiteLLM UI ──
UI_USERNAME="admin"
UI_PASSWORD="your-secure-password"

# ── Langfuse (auto-configured, see setup below) ──
# LANGFUSE_PUBLIC_KEY=""   # Set after Langfuse setup
# LANGFUSE_SECRET_KEY=""   # Set after Langfuse setup
```

### LiteLLM Config (config.docker.yaml)

The proxy is configured via `config.docker.yaml` mounted into the container:

```yaml
model_list:
  # Chat model via OpenRouter
  - model_name: gpt-4o
    litellm_params:
      model: openrouter/openai/gpt-4o
      api_key: os.environ/OPENROUTER_API_KEY

  # In-process embeddings (pre-baked into image)
  - model_name: bge-small
    litellm_params:
      model: fastembed/BAAI/bge-small-en-v1.5

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  ui_access_mode: admin

litellm_settings:
  cache: true
  cache_params:
    type: "qdrant-semantic"
    qdrant_api_base: "http://qdrant:6333"
    qdrant_collection_name: "litellm_cache"
    qdrant_semantic_cache_embedding_model: "bge-small"
    similarity_threshold: 0.8
    ttl: 604800          # 7 days

  success_callback: ["langfuse"]
  failure_callback: ["langfuse"]
```

## 📊 Langfuse Observability Setup

Langfuse is included in the stack and tracks **every LLM call** through LiteLLM — tokens, costs, latency, model usage.

### Step 1: Create Langfuse Account

```bash
# Start the stack first
docker compose -f docker-compose.enterprise.yml up -d

# Wait ~90s for all services to initialize, then:
open http://localhost:3001
```

Create your admin account in the Langfuse UI.

### Step 2: Generate API Keys

1. Go to **Settings → API Keys** in Langfuse UI
2. Create a new key pair
3. Copy the `pk-lf-...` (public) and `sk-lf-...` (secret) keys

### Step 3: Connect LiteLLM to Langfuse

Add the keys to your `.env`:

```bash
LANGFUSE_PUBLIC_KEY="pk-lf-your-key"
LANGFUSE_SECRET_KEY="sk-lf-your-secret"
```

Then restart LiteLLM to pick up the new keys:

```bash
docker compose -f docker-compose.enterprise.yml up -d --force-recreate litellm
```

### Step 4: Verify

```bash
# Make a test call
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-master-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello!"}]}'

# Check Langfuse UI — trace should appear within seconds
open http://localhost:3001
```

### What Langfuse Tracks

- **Every request** — input, output, model, tokens, latency
- **Cost tracking** — per-model, per-key, per-team spend
- **Error traces** — failed calls with full stack context
- **Dashboards** — usage over time, cost trends, model comparison

## ⚡ Semantic Caching

Qdrant semantic cache gives **10-100x faster responses** for repeated/similar queries while saving LLM costs.

### How It Works

1. First request → LLM is called → response cached as vector embedding in Qdrant
2. Subsequent similar requests → Qdrant finds matching response → returned instantly
3. Cache hit bypasses the LLM entirely — zero cost, near-zero latency

### Verify Caching

```bash
# First call — hits LLM (slow)
time curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-master-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"What is the capital of France?"}]}' | jq -r '.choices[0].message.content'

# Same/similar query — cache hit (fast)
time curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-master-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"What is France's capital?"}]}' | jq -r '.choices[0].message.content'
```

### Cache Configuration

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: "qdrant-semantic"
    qdrant_api_base: "http://qdrant:6333"
    qdrant_semantic_cache_embedding_model: "bge-small"
    similarity_threshold: 0.8    # 0.0-1.0, higher = stricter match
    ttl: 604800                   # Cache TTL in seconds (7 days)
    supported_call_types:
      - "completion"
      - "acompletion"
      - "embedding"
      - "aembedding"
```

## 🔌 Usage

### OpenAI-Compatible API

All requests use the standard OpenAI format — just point your base URL:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="sk-your-master-key"
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Embeddings

```python
response = client.embeddings.create(
    model="bge-small",
    input="What is the capital of France?"
)
print(response.data[0].embedding[:5])  # 384-dim vector
```

### cURL

```bash
# Chat completion
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-master-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello!"}]}'

# Embedding
curl http://localhost:4000/v1/embeddings \
  -H "Authorization: Bearer sk-your-master-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-small","input":"Hello world"}'

# Health check
curl http://localhost:4000/health/liveliness
curl http://localhost:4000/health/readiness
```

### Using from Other Frameworks

Works with any OpenAI-compatible client:

```javascript
// Node.js / Vercel AI SDK
import { createOpenAI } from '@ai-sdk/openai';
const litellm = createOpenAI({
  baseURL: 'http://localhost:4000/v1',
  apiKey: 'sk-your-master-key',
});
```

```yaml
# LangChain
llm:
  _type: openai
  openai_api_base: "http://localhost:4000/v1"
  openai_api_key: "sk-your-master-key"
  model_name: "gpt-4o"
```

## 🔑 API Key Management

Create virtual keys with budgets and rate limits via the LiteLLM UI or API:

```bash
# Create a key with $10 budget
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer sk-your-master-key" \
  -H "Content-Type: application/json" \
  -d '{"max_budget": 10.0, "key_alias": "my-app-key"}'

# Response: {"key": "sk-litellm-...", "max_budget": 10.0, ...}
```

Use the generated key in your apps instead of the master key.

## 🛠️ Management Commands

```bash
# Start all services
docker compose -f docker-compose.enterprise.yml up -d

# Stop all services
docker compose -f docker-compose.enterprise.yml down

# View logs
docker compose -f docker-compose.enterprise.yml logs -f litellm

# Restart single service
docker compose -f docker-compose.enterprise.yml restart litellm

# Pull latest image and restart
docker compose -f docker-compose.enterprise.yml pull litellm
docker compose -f docker-compose.enterprise.yml up -d --force-recreate litellm

# Check all service health
docker compose -f docker-compose.enterprise.yml ps
```

## 🔒 Production Notes

Before deploying to production:

1. **Change all default passwords** in `docker-compose.enterprise.yml`:
   - `POSTGRES_PASSWORD` (both databases)
   - `NEXTAUTH_SECRET` and `SALT` (Langfuse)
   - `MINIO_ROOT_PASSWORD`
   - `REDIS` password
   - `CLICKHOUSE_PASSWORD`
   - `ENCRYPTION_KEY` (Langfuse, generate a new 32-byte hex key)

2. **Set `LITELLM_MASTER_KEY`** to a strong random key

3. **Add TLS** — put a reverse proxy (nginx/caddy) in front

4. **Restrict port exposure** — don't expose DB/Redis/ClickHouse ports publicly

5. **Set up backups** for PostgreSQL volumes

## 🐛 Troubleshooting

### Container won't start
```bash
docker compose -f docker-compose.enterprise.yml logs litellm
```

### Langfuse traces not appearing
```bash
# Verify Langfuse is healthy
curl http://localhost:3001

# Check callbacks are registered
curl http://localhost:4000/health \
  -H "Authorization: Bearer sk-your-master-key"
# Look for "langfuse" in success_callback
```

### Semantic cache not working
```bash
# Verify Qdrant is healthy
curl http://localhost:6335/collections/litellm_cache
```

### Port conflicts
Edit ports in `docker-compose.enterprise.yml`:
- LiteLLM: `4000:4000` → `YOUR_PORT:4000`
- Langfuse: `3001:3000` → `YOUR_PORT:3000`
- Qdrant: `6335:6333` → `YOUR_PORT:6333`

## 📄 License

This fork is based on [BerriAI/litellm](https://github.com/BerriAI/litellm) (MIT License).
Enterprise features are proprietary additions.
