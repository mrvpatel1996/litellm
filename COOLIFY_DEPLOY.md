# 🚀 LiteLLM Enterprise — Complete Coolify Deployment Guide

Deploy the full 10-container stack (LiteLLM + Qdrant + SearXNG + Langfuse + ClickHouse + Redis + MinIO + 2× PostgreSQL) to any VPS via Coolify.

---

## 📋 Prerequisites

### What You Need

| Item | Details |
|------|---------|
| **VPS** | 4GB+ RAM, 2+ vCPU, 40GB+ disk (Ubuntu 22.04/24.04 recommended) |
| **Domain** | A domain name with DNS access (e.g., `yourdomain.com`) |
| **Coolify** | Installed on the VPS (self-hosted PaaS) |
| **GitHub Repo** | Your fork pushed to GitHub (mrvpatel1996/litellm) |
| **OpenRouter Key** | API key from [openrouter.ai](https://openrouter.ai) |

### DNS Records to Create

Create A records pointing to your VPS IP:

```
litellm.yourdomain.com     → VPS_IP   (LiteLLM UI + API)
langfuse.yourdomain.com    → VPS_IP   (Langfuse observability)
search.yourdomain.com      → VPS_IP   (SearXNG search UI)
```

---

## Step 1: Install Coolify on Your VPS

SSH into your VPS and run:

```bash
# One-command install (takes ~5 minutes)
curl -fsSL https://cdn.coollabs.io/coolify/install.sh | bash
```

This installs Docker, Coolify, and all dependencies. Once done:

1. Open `http://YOUR_VPS_IP:8000` in your browser
2. Create admin account
3. You're in the Coolify dashboard

---

## Step 2: Push Code to GitHub

Your repo is already at `github.com/mrvpatel1996/litellm` (branch `enterprise-features`).

Make sure these new files are committed and pushed:

```bash
cd ~/Desktop/litellm

# New files for Coolify deployment
git add docker-compose.coolify.yml config.coolify.yaml env.coolify.example searxng/
git commit -m "feat: add Coolify deployment files + SearXNG search"
git push origin enterprise-features
```

**Files Coolify needs from the repo:**
- `docker-compose.coolify.yml` — the production compose file
- `config.coolify.yaml` — LiteLLM proxy config (mounted into container)
- `Dockerfile.enterprise` — for building the LiteLLM image
- `searxng/settings.yml` — SearXNG configuration
- `docker/entrypoint.enterprise.sh` — startup script

---

## Step 3: Create a Coolify Project

1. In Coolify dashboard → **Projects** → **+ New Project**
2. Name it: `LiteLLM Enterprise`
3. Select your server (the VPS where Coolify runs)
4. Choose project type: **Docker Compose**

---

## Step 4: Connect Your GitHub Repo

### Option A: Use Coolify's GitHub Integration (Recommended)

1. In the project → **Source** → **GitHub**
2. Authorize Coolify to access your GitHub account
3. Select repository: `mrvpatel1996/litellm`
4. Select branch: `enterprise-features`
5. Set the compose file path: `docker-compose.coolify.yml`

### Option B: Use a Public Repo URL

If your repo is public:
1. In the project → **Source** → **Public Repository**
2. URL: `https://github.com/mrvpatel1996/litellm`
3. Branch: `enterprise-features`
4. Compose file: `docker-compose.coolify.yml`

---

## Step 5: Configure Environment Variables

In Coolify → your project → **Environment Variables** tab.

### Generate Secrets First

Run these locally and save the outputs:

```bash
# Generate all required secrets
echo "LITELLM_MASTER_KEY=sk-$(openssl rand -hex 24)"
echo "DB_PASSWORD=$(openssl rand -hex 16)"
echo "LANGFUSE_DB_PASSWORD=$(openssl rand -hex 16)"
echo "LANGFUSE_NEXTAUTH_SECRET=$(openssl rand -hex 32)"
echo "LANGFUSE_SALT=$(openssl rand -hex 32)"
echo "LANGFUSE_ENCRYPTION_KEY=$(openssl rand -hex 32)"
echo "CLICKHOUSE_PASSWORD=$(openssl rand -hex 16)"
echo "REDIS_PASSWORD=$(openssl rand -hex 16)"
echo "MINIO_PASSWORD=$(openssl rand -hex 16)"
echo "SEARXNG_SECRET=$(openssl rand -hex 32)"
echo "UI_PASSWORD=$(openssl rand -base64 16)"
```

### Paste into Coolify

Copy-paste all variables into the Coolify environment variables editor. See `env.coolify.example` for the template. Replace these:

```env
# ── Core ──
LITELLM_MASTER_KEY=sk-YOUR_GENERATED_KEY
OPENROUTER_API_KEY=sk-or-YOUR_OPENROUTER_KEY
UI_USERNAME=admin
UI_PASSWORD=YOUR_GENERATED_PASSWORD

# ── Database Passwords ──
DB_PASSWORD=YOUR_DB_PASSWORD
LANGFUSE_DB_PASSWORD=YOUR_LANGFUSE_DB_PASSWORD

# ── Langfuse (set NEXTAUTH_URL to your actual domain) ──
LANGFUSE_NEXTAUTH_URL=https://langfuse.yourdomain.com
LANGFUSE_NEXTAUTH_SECRET=YOUR_64_HEX_SECRET
LANGFUSE_SALT=YOUR_64_HEX_SALT
LANGFUSE_ENCRYPTION_KEY=YOUR_64_HEX_ENCRYPTION_KEY
# These two are EMPTY for first deploy — fill after Langfuse setup
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=

# ── ClickHouse ──
CLICKHOUSE_PASSWORD=YOUR_CLICKHOUSE_PASSWORD

# ── Redis ──
REDIS_PASSWORD=YOUR_REDIS_PASSWORD

# ── MinIO ──
MINIO_PASSWORD=YOUR_MINIO_PASSWORD

# ── SearXNG ──
SEARXNG_BASE_URL=https://search.yourdomain.com
SEARXNG_SECRET=YOUR_SEARXNG_SECRET
```

---

## Step 6: Configure Domains (Coolify Proxy)

Coolify uses Traefik/Caddy as a reverse proxy for automatic SSL.

### For the LiteLLM Service

1. In Coolify project → click on the `litellm` service
2. Go to **Domains** tab
3. Set domain: `https://litellm.yourdomain.com`
4. Port: `4000`
5. Coolify auto-provisions Let's Encrypt SSL certificate

### For the Langfuse Service

1. Click on `langfuse-web` service
2. **Domains** → `https://langfuse.yourdomain.com`
3. Port: `3000`

### For the SearXNG Service

1. Click on `searxng` service
2. **Domains** → `https://search.yourdomain.com`
3. Port: `8080`

> **Note:** Qdrant, PostgreSQL, ClickHouse, Redis, MinIO stay internal — no public domains needed.

---

## Step 7: First Deploy

1. Click **Deploy** in Coolify
2. Wait ~3-5 minutes:
   - LiteLLM image builds or pulls from GHCR
   - All services start in dependency order
   - Health checks pass
3. Coolify shows all services as **Running**

### Expected Startup Order

```
PostgreSQL → Qdrant → SearXNG → ClickHouse → Redis → MinIO
    → Langfuse DB ready
        → Langfuse Web + Worker start
            → LiteLLM starts (waits for DB + Qdrant)
```

---

## Step 8: Post-Deploy — Langfuse Setup

Langfuse needs first-time configuration to generate API keys.

### 8a. Create Admin Account

```bash
curl -X POST https://langfuse.yourdomain.com/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Admin",
    "email": "admin@yourdomain.com",
    "password": "YourStrongPassword123!"
  }'
```

### 8b. Get API Keys via Browser

1. Open `https://langfuse.yourdomain.com`
2. Sign in with your admin account
3. **Create Organization** → name it (e.g., "Enterprise")
4. **Create Project** → name it (e.g., "Production")
5. On the "Log your first trace" page, you'll see:
   - **Secret Key**: `sk-lf-xxxxxxxxx`
   - **Public Key**: `pk-lf-xxxxxxxxx`

### 8c. Update Coolify Environment Variables

1. Go back to Coolify → Environment Variables
2. Set:
   ```env
   LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxx
   LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxx
   ```
3. **Redeploy** the `litellm` service (not full stack — just `litellm`)

### 8d. Verify Langfuse Integration

```bash
# Check LiteLLM registered the callback
curl -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  https://litellm.yourdomain.com/health/readiness

# Should show "langfuse" in success_callbacks
```

Make an LLM call, then check Langfuse UI → Traces — you should see it appear within seconds.

---

## Step 9: Verify Everything

Run this checklist after deployment:

```bash
# Set your domain
export LITELLM_URL="https://litellm.yourdomain.com"
export MASTER_KEY="sk-your-master-key"

# 1. LiteLLM health
curl -sf $LITELLM_URL/health/liveliness
# Expected: "I'm alive!"

# 2. Database connected
curl -sf $LITELLM_URL/health/readiness | jq .
# Expected: {"status":"healthy","db":"connected"}

# 3. Models loaded
curl -sf -H "Authorization: Bearer $MASTER_KEY" $LITELLM_URL/v1/models | jq '.data[].id'
# Expected: ["gpt-4o","bge-small"]

# 4. SearXNG search via proxy
curl -sf -H "Authorization: Bearer $MASTER_KEY" \
  $LITELLM_URL/v1/search/searxng-search \
  -H "Content-Type: application/json" \
  -d '{"query":"latest AI news","max_results":5}' | jq '.results[0].title'
# Expected: a news headline

# 5. Chat completion (uses OpenRouter)
curl -sf -H "Authorization: Bearer $MASTER_KEY" \
  $LITELLM_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello!"}]}' | jq '.choices[0].message.content'

# 6. Enterprise modules
curl -sf -H "Authorization: Bearer $MASTER_KEY" \
  $LITELLM_URL/enterprise/health | jq .
```

---

## Step 10: Register Vector Store (One-Time)

After first deploy, register the Qdrant vector store in the database:

```bash
curl -X POST $LITELLM_URL/vector_store/new \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "vector_store_id": "countries",
    "custom_llm_provider": "qdrant",
    "vector_store_name": "Countries",
    "litellm_params": {
      "api_base": "http://qdrant:6333",
      "litellm_embedding_model": "bge-small",
      "qdrant_distance": "Cosine",
      "qdrant_vector_size": 384,
      "qdrant_text_field": "text"
    }
  }'
```

---

## Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │       Coolify (Traefik)          │
                    │   Automatic SSL + Reverse Proxy  │
                    └──────┬──────────┬──────────┬─────┘
                           │          │          │
                  ┌────────▼──┐ ┌─────▼─────┐ ┌──▼──────┐
                  │  LiteLLM  │ │ Langfuse  │ │ SearXNG │
                  │  :4000    │ │  :3000    │ │  :8080  │
                  └─────┬─────┘ └─────┬─────┘ └─────────┘
                        │             │
           ┌────────────┼─────────────┼──────────────┐
           │            │             │              │
     ┌─────▼────┐ ┌────▼───┐ ┌───────▼───────┐ ┌───▼────────┐
     │ Qdrant   │ │  PG DB │ │ ClickHouse    │ │  Redis     │
     │ :6333    │ │ :5432  │ │ (analytics)   │ │ (queue)    │
     │ Vector   │ │ LiteLLM│ └───────────────┘ └────────────┘
     │ + Cache  │ └────────┘
     └──────────┘    ┌──────────────┐
                     │  MinIO :9000 │
                     │  (S3 storage)│
                     └──────────────┘
                     ┌──────────────┐
                     │  Langfuse PG │
                     │  :5432       │
                     └──────────────┘
```

## Service Summary

| Service | Internal Port | Public Domain | Purpose |
|---------|--------------|---------------|---------|
| LiteLLM | 4000 | `litellm.yourdomain.com` | AI gateway UI + API |
| Langfuse | 3000 | `langfuse.yourdomain.com` | Observability + tracing |
| SearXNG | 8080 | `search.yourdomain.com` | Web search engine |
| Qdrant | 6333 | — (internal) | Vector store + semantic cache |
| PostgreSQL (LiteLLM) | 5432 | — (internal) | Proxy DB |
| PostgreSQL (Langfuse) | 5432 | — (internal) | Langfuse DB |
| ClickHouse | 8123 | — (internal) | OLAP analytics |
| Redis | 6379 | — (internal) | Cache + job queue |
| MinIO | 9000/9001 | — (internal) | S3-compatible storage |

---

## Resource Requirements

| Component | RAM | CPU | Disk |
|-----------|-----|-----|------|
| LiteLLM | ~500MB | 0.5 | 1GB |
| Qdrant | ~200MB | 0.2 | 5GB+ |
| SearXNG | ~150MB | 0.2 | 0.5GB |
| Langfuse (web+worker) | ~400MB | 0.3 | 1GB |
| ClickHouse | ~500MB | 0.5 | 5GB+ |
| Redis | ~50MB | 0.1 | 0.5GB |
| MinIO | ~100MB | 0.1 | 5GB+ |
| 2× PostgreSQL | ~200MB | 0.2 | 2GB |
| **Total** | **~2.1GB** | **~2.0** | **~20GB+** |

**Recommended VPS:** 4GB RAM, 2 vCPU, 40GB SSD

---

## Troubleshooting

### Container won't start
```bash
# Check logs for the failing service
docker logs <container_name> --tail 50

# In Coolify: Service → Logs tab
```

### Langfuse shows blank page
- `LANGFUSE_NEXTAUTH_URL` must match the exact domain Coolify assigned (with `https://`)
- Check `LANGFUSE_ENCRYPTION_KEY` is exactly 64 hex characters

### LiteLLM shows no Langfuse callbacks
- `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` must be filled in Coolify env vars
- Redeploy LiteLLM after adding them (not just restart)
- Verify: `curl $LITELLM_URL/health/readiness` → look for `langfuse` in callbacks

### SearXNG returns HTML instead of JSON
- The `searxng/settings.yml` must have `formats: [html, json]`
- Check the volume mount is correct in Coolify

### Database connection errors
- `DATABASE_URL` should be `postgresql://litellm:***@db:5432/litellm`
- The password must match `DB_PASSWORD` env var
- On first deploy, LiteLLM auto-creates tables via `--use_prisma_db_push`

### ClickHouse credential mismatch
- If you change `CLICKHOUSE_PASSWORD`, you must delete the volume:
  ```bash
  docker volume rm litellm_litellm_enterprise_clickhouse
  ```
  Then redeploy. Old volumes cache stale credentials.

### Docker pull hangs
- Remove `"credsStore": "desktop"` from `~/.docker/config.json` on the VPS:
  ```bash
  python3 -c "
  import json
  p = '/root/.docker/config.json'
  c = json.load(open(p))
  c.pop('credsStore', None)
  json.dump(c, open(p, 'w'), indent=2)
  "
  ```

---

## Updating / Redeploying

### After code changes

```bash
# Push to GitHub
cd ~/Desktop/litellm
git add -A && git commit -m "changes" && git push

# In Coolify: Click "Redeploy" — it pulls latest code and rebuilds
```

### After adding new models or config changes

1. Edit `config.coolify.yaml` locally
2. Commit + push to GitHub
3. In Coolify → Redeploy (the config file is mounted from the repo)

### Forcing image rebuild

In Coolify → Service → Settings → toggle **Build Pack** to "Dockerfile" → set `Dockerfile.enterprise` as the Dockerfile location → Redeploy.

---

## Backup Strategy

### Database backups (daily cron)

```bash
# Add to VPS crontab
0 3 * * * docker exec litellm-enterprise-db pg_dump -U litellm litellm | gzip > /backups/litellm-$(date +\%Y\%m\%d).sql.gz
0 3 * * * docker exec litellm-enterprise-langfuse-db pg_dump -U langfuse langfuse | gzip > /backups/langfuse-$(date +\%m\%d).sql.gz
```

### Qdrant snapshots

```bash
# Create snapshot
curl -X POST http://localhost:6333/snapshots
```

---

## Quick Reference: API Endpoints

```bash
# All endpoints available at https://litellm.yourdomain.com

# Chat
POST /v1/chat/completions

# Embeddings
POST /v1/embeddings

# Web Search
POST /v1/search/searxng-search

# Vector Store Search
POST /v1/vector_stores/{id}/search

# Health
GET  /health/liveliness    (no auth)
GET  /health/readiness     (no auth)
GET  /health               (auth required)

# Enterprise
GET  /enterprise/health    (auth required)

# Models
GET  /v1/models            (auth required)
GET  /model/info           (auth required)

# UI
GET  /ui
```
