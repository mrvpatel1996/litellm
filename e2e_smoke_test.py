#!/usr/bin/env python3
"""
LiteLLM Enterprise — End-to-End Smoke Test

Tests: proxy health, caching, embeddings, enterprise endpoints, Langfuse integration.

Usage:
  python3 e2e_smoke_test.py                  # Run all tests
  python3 e2e_smoke_test.py --url http://localhost:4000
  python3 e2e_smoke_test.py --skip-langfuse  # Skip Langfuse checks (no keys)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import requests

# ─── Config ──────────────────────────────────────────────────
BASE_URL = "http://localhost:4000"
LITELLM_DIR = Path(__file__).parent


def load_api_key():
    """Load master key from .env."""
    env_file = LITELLM_DIR / ".env"
    if not env_file.exists():
        print("❌ .env file not found")
        sys.exit(1)
    for line in env_file.read_text().split("\n"):
        if line.startswith("LITELLM_MASTER_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    print("❌ LITELLM_MASTER_KEY not in .env")
    sys.exit(1)


def load_langfuse_keys():
    """Load Langfuse keys from .env."""
    env_file = LITELLM_DIR / ".env"
    keys = {"public": "", "secret": "", "host": "https://cloud.langfuse.com"}
    for line in env_file.read_text().split("\n"):
        if line.startswith("LANGFUSE_PUBLIC_KEY="):
            keys["public"] = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("LANGFUSE_SECRET_KEY=") and "UPSTREAM" not in line:
            keys["secret"] = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("LANGFUSE_HOST="):
            keys["host"] = line.split("=", 1)[1].strip().strip('"')
    return keys


# ─── Test Helpers ─────────────────────────────────────────────
class TestRunner:
    def __init__(self, base_url, api_key, skip_langfuse=False):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.skip_langfuse = skip_langfuse
        self.passed = 0
        self.failed = 0
        self.results = []

    def _record(self, name, passed, detail=""):
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results.append((name, passed, detail))
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  {status} — {name}" + (f" ({detail})" if detail else ""))

    def test(self, name, condition, detail=""):
        self._record(name, condition, detail)

    # ─── Tests ────────────────────────────────────────────
    def test_liveliness(self):
        try:
            r = requests.get(f"{self.base_url}/health/liveliness", timeout=10)
            self.test("Proxy liveliness", r.status_code == 200 and "alive" in r.text.lower(), f"status={r.status_code}")
        except Exception as e:
            self.test("Proxy liveliness", False, str(e))

    def test_health(self):
        try:
            r = requests.get(f"{self.base_url}/health", headers=self.headers, timeout=15)
            self.test("Health endpoint", r.status_code == 200, f"status={r.status_code}")
            if r.status_code == 200:
                data = r.json()
                models = [e.get("model", "") for e in data.get("healthy_endpoints", [])]
                self.test("gpt-4o model registered", any("gpt-4o" in m for m in models))
        except Exception as e:
            self.test("Health endpoint", False, str(e))

    def test_models_list(self):
        try:
            r = requests.get(f"{self.base_url}/v1/models", headers=self.headers, timeout=10)
            self.test("Models list", r.status_code == 200, f"status={r.status_code}")
            if r.status_code == 200:
                data = r.json()
                model_ids = [m["id"] for m in data.get("data", [])]
                self.test("gpt-4o in models", "gpt-4o" in model_ids, f"models={model_ids}")
                self.test("bge-small in models", "bge-small" in model_ids)
        except Exception as e:
            self.test("Models list", False, str(e))

    def test_chat_completion(self):
        try:
            t0 = time.time()
            r = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
                    "max_tokens": 10,
                },
                timeout=60,
            )
            elapsed = time.time() - t0
            self.test("Chat completion (gpt-4o)", r.status_code == 200, f"status={r.status_code}, time={elapsed:.2f}s")
            if r.status_code == 200:
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                self.test("Response contains answer", "4" in content, f'response="{content[:50]}"')
                return data
        except Exception as e:
            self.test("Chat completion (gpt-4o)", False, str(e))
        return None

    def test_caching(self, first_response):
        """Test that the same prompt gets a cache hit (much faster)."""
        if first_response is None:
            self.test("Caching (cache hit)", False, "No first response to compare")
            return
        try:
            t0 = time.time()
            r = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
                    "max_tokens": 10,
                },
                timeout=60,
            )
            elapsed = time.time() - t0
            is_cache_hit = elapsed < 0.5  # Cache hits are < 100ms typically
            self.test("Caching (cache hit)", r.status_code == 200 and is_cache_hit,
                       f"time={elapsed:.3f}s {'(cached!)' if is_cache_hit else '(miss)'}")
        except Exception as e:
            self.test("Caching (cache hit)", False, str(e))

    def test_embeddings(self):
        try:
            r = requests.post(
                f"{self.base_url}/v1/embeddings",
                headers=self.headers,
                json={"model": "bge-small", "input": "test embedding for smoke test"},
                timeout=30,
            )
            self.test("Embedding (bge-small)", r.status_code == 200, f"status={r.status_code}")
            if r.status_code == 200:
                data = r.json()
                dim = len(data["data"][0]["embedding"])
                self.test("Embedding dimension", dim == 384, f"dim={dim}")
        except Exception as e:
            self.test("Embedding (bge-small)", False, str(e))

    def test_enterprise_health(self):
        try:
            r = requests.get(f"{self.base_url}/enterprise/health", headers=self.headers, timeout=10)
            self.test("Enterprise health", r.status_code == 200, f"status={r.status_code}")
            if r.status_code == 200:
                data = r.json()
                features = data.get("features", {})
                for feat, expected in [
                    ("saml", True), ("rbac", True), ("audit", True),
                    ("pricing", True), ("multi_tenancy", True), ("vectordb", True)
                ]:
                    self.test(f"Enterprise {feat}", features.get(feat) == expected, f"value={features.get(feat)}")
        except Exception as e:
            self.test("Enterprise health", False, str(e))

    def test_enterprise_roles(self):
        try:
            r = requests.get(f"{self.base_url}/enterprise/roles", headers=self.headers, timeout=10)
            self.test("Enterprise roles", r.status_code == 200, f"status={r.status_code}")
            if r.status_code == 200:
                data = r.json()
                roles = data.get("roles", [])
                self.test("Roles count", len(roles) >= 5, f"count={len(roles)}")
        except Exception as e:
            self.test("Enterprise roles", False, str(e))

    def test_enterprise_pricing(self):
        try:
            r = requests.get(f"{self.base_url}/enterprise/pricing/rules", headers=self.headers, timeout=10)
            self.test("Enterprise pricing", r.status_code == 200, f"status={r.status_code}")
        except Exception as e:
            self.test("Enterprise pricing", False, str(e))

    def test_langfuse_config(self):
        if self.skip_langfuse:
            self.test("Langfuse (skipped)", True, "user requested skip")
            return
        keys = load_langfuse_keys()
        has_keys = bool(keys["public"] and keys["secret"])
        self.test("Langfuse keys configured", has_keys,
                   f"public={keys['public'][:4]}..." if keys["public"] else "EMPTY — add to .env")
        if not has_keys:
            print("  ⚠️  Langfuse callbacks registered but no keys set.")
            print("     Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to .env")

    def test_vector_store(self):
        try:
            # Check if Qdrant is reachable
            r = requests.get("http://localhost:6333/collections", timeout=5)
            self.test("Qdrant reachable", r.status_code == 200, f"status={r.status_code}")
            if r.status_code == 200:
                collections = r.json().get("result", {}).get("collections", [])
                names = [c["name"] for c in collections]
                self.test("litellm_cache collection", "litellm_cache" in names, f"collections={names}")
        except Exception as e:
            self.test("Qdrant reachable", False, str(e))

    # ─── Main ─────────────────────────────────────────────
    def run_all(self):
        print(f"\n{'='*60}")
        print(f"  LiteLLM Enterprise — Smoke Test")
        print(f"  URL: {self.base_url}")
        print(f"{'='*60}\n")

        print("📡 Connectivity")
        self.test_liveliness()
        self.test_health()

        print("\n📋 Models")
        self.test_models_list()

        print("\n💬 Chat + Caching")
        first = self.test_chat_completion()
        self.test_caching(first)

        print("\n🔢 Embeddings")
        self.test_embeddings()

        print("\n🏢 Enterprise Features")
        self.test_enterprise_health()
        self.test_enterprise_roles()
        self.test_enterprise_pricing()

        print("\n📊 Observability")
        self.test_langfuse_config()

        print("\n🗄️  Vector Store + Cache")
        self.test_vector_store()

        # Summary
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.failed > 0:
            print(f"\n  Failed tests:")
            for name, passed, detail in self.results:
                if not passed:
                    print(f"    ❌ {name}: {detail}")
        print(f"{'='*60}\n")
        return self.failed == 0


def main():
    parser = argparse.ArgumentParser(description="LiteLLM Enterprise Smoke Test")
    parser.add_argument("--url", default=BASE_URL, help="Proxy URL")
    parser.add_argument("--skip-langfuse", action="store_true", help="Skip Langfuse checks")
    args = parser.parse_args()

    api_key = load_api_key()
    runner = TestRunner(args.url, api_key, skip_langfuse=args.skip_langfuse)
    success = runner.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
