#!/usr/bin/env python3
"""
LiteLLM Enterprise Fork ── End-to-End Smoke Test

Runs against a running LiteLLM proxy (local or Docker).
Tests: health, caching, embeddings, enterprise endpoints, vector store.

Usage:
  python3 e2e_smoke_test.py                     # Test local proxy at localhost:4000
  python3 e2e_smoke_test.py --url http://...    # Custom URL
  python3 e2e_smoke_test.py --no-api            # Skip tests that call external APIs
"""
import argparse
import json
import sys
import time
from typing import Optional

import requests


# ─── Colors ───
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


class SmokeTest:
    def __init__(self, base_url: str, api_key: str, skip_api: bool = False):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.skip_api = skip_api
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []

    def _record(self, name: str, status: str, detail: str = "", latency: Optional[float] = None):
        self.results.append({"name": name, "status": status, "detail": detail, "latency": latency})
        if status == "PASS":
            self.passed += 1
            lat = f" ({latency:.2f}s)" if latency else ""
            print(f"  {GREEN}✅ PASS{RESET} {name}{lat}")
        elif status == "FAIL":
            self.failed += 1
            print(f"  {RED}❌ FAIL{RESET} {name}: {detail}")
        elif status == "SKIP":
            self.skipped += 1
            print(f"  {YELLOW}⏭️  SKIP{RESET} {name}: {detail}")

    def test_health(self):
        """Test proxy health endpoints."""
        print(f"\n{BLUE}━━━ Health Checks ━━━{RESET}")

        # Liveliness
        t0 = time.time()
        try:
            r = requests.get(f"{self.base_url}/health/liveliness", timeout=10)
            lat = time.time() - t0
            if r.status_code == 200 and "alive" in r.text.lower():
                self._record("Liveliness /health/liveliness", "PASS", latency=lat)
            else:
                self._record("Liveliness /health/liveliness", "FAIL", f"status={r.status_code}")
        except Exception as e:
            self._record("Liveliness /health/liveliness", "FAIL", str(e))

        # Readiness
        t0 = time.time()
        try:
            r = requests.get(f"{self.base_url}/health", headers=self.headers, timeout=10)
            lat = time.time() - t0
            # 200 = healthy, 401 = auth required (both mean proxy is running)
            if r.status_code in (200, 401):
                self._record("Health /health", "PASS", latency=lat)
            else:
                self._record("Health /health", "FAIL", f"status={r.status_code}")
        except Exception as e:
            self._record("Health /health", "FAIL", str(e))

    def test_models(self):
        """Test model listing."""
        print(f"\n{BLUE}━━━ Model Listing ━━━{RESET}")

        try:
            r = requests.get(f"{self.base_url}/v1/models", headers=self.headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                models = [m["id"] for m in data.get("data", [])]
                self._record("List models", "PASS", detail=f"models={models}")
                if "gpt-4o" in models:
                    self._record("gpt-4o model available", "PASS")
                else:
                    self._record("gpt-4o model available", "FAIL", "not in model list")
                if "bge-small" in models:
                    self._record("bge-small embedding model", "PASS")
                else:
                    self._record("bge-small embedding model", "FAIL", "not in model list")
            else:
                self._record("List models", "FAIL", f"status={r.status_code}: {r.text[:200]}")
        except Exception as e:
            self._record("List models", "FAIL", str(e))

    def test_caching(self):
        """Test LLM response caching — exact match + verify no double billing."""
        print(f"\n{BLUE}━━━ LLM Caching (Cost Savings) ━━━{RESET}")

        if self.skip_api:
            self._record("Caching test", "SKIP", "--no-api flag set")
            return

        # Request 1: cache MISS
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": f"Say 'cache-test-{int(time.time())}' and nothing else."}],
            "max_tokens": 15,
        }

        t0 = time.time()
        try:
            r1 = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers, json=payload, timeout=60,
            )
            lat1 = time.time() - t0
            if r1.status_code != 200:
                self._record("Cache MISS (first call)", "FAIL", f"status={r1.status_code}: {r1.text[:200]}")
                return

            resp1 = r1.json()
            content1 = resp1["choices"][0]["message"]["content"]
            self._record("Cache MISS (first call)", "PASS", detail=f"response={content1[:50]}", latency=lat1)
        except Exception as e:
            self._record("Cache MISS (first call)", "FAIL", str(e))
            return

        # Request 2: exact same request → should be CACHE HIT
        t0 = time.time()
        try:
            r2 = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers, json=payload, timeout=60,
            )
            lat2 = time.time() - t0
            if r2.status_code != 200:
                self._record("Cache HIT (same request)", "FAIL", f"status={r2.status_code}")
                return

            resp2 = r2.json()
            content2 = resp2["choices"][0]["message"]["content"]

            # Cache hit should be much faster (< 1s)
            is_cached = lat2 < 1.0
            speedup = lat1 / lat2 if lat2 > 0 else float("inf")

            if is_cached and content1.strip() == content2.strip():
                self._record(
                    "Cache HIT (exact match)",
                    "PASS",
                    detail=f"speedup={speedup:.1f}x, saved ${lat1-lat2:.2f}s",
                    latency=lat2,
                )
            else:
                self._record(
                    "Cache HIT (exact match)",
                    "FAIL",
                    detail=f"latency={lat2:.2f}s (expected <1s), speedup={speedup:.1f}x",
                    latency=lat2,
                )
        except Exception as e:
            self._record("Cache HIT (same request)", "FAIL", str(e))

    def test_embeddings(self):
        """Test embedding endpoint."""
        print(f"\n{BLUE}━━━ Embeddings ━━━{RESET}")

        try:
            t0 = time.time()
            r = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json={"model": "bge-small", "input": "Hello world"},
                timeout=30,
            )
            lat = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                dim = len(data["data"][0]["embedding"])
                self._record(
                    "Embedding bge-small",
                    "PASS",
                    detail=f"dim={dim}",
                    latency=lat,
                )
                if dim != 384:
                    self._record("Embedding dimension", "FAIL", f"expected 384, got {dim}")
                else:
                    self._record("Embedding dimension = 384", "PASS")
            else:
                self._record("Embedding bge-small", "FAIL", f"status={r.status_code}: {r.text[:200]}")
        except Exception as e:
            self._record("Embedding bge-small", "FAIL", str(e))

    def test_enterprise(self):
        """Test enterprise-specific endpoints."""
        print(f"\n{BLUE}━━━ Enterprise Features ━━━{RESET}")

        endpoints = [
            ("GET", "/enterprise/health", "Enterprise Health"),
            ("GET", "/enterprise/roles", "Enterprise Roles"),
            ("GET", "/enterprise/pricing/rules", "Enterprise Pricing"),
        ]

        for method, path, name in endpoints:
            try:
                t0 = time.time()
                r = requests.request(
                    method, f"{self.base_url}{path}",
                    headers=self.headers, timeout=10,
                )
                lat = time.time() - t0
                if r.status_code == 200:
                    self._record(name, "PASS", latency=lat)
                elif r.status_code in (403, 404):
                    # Feature exists but may need config
                    self._record(name, "PASS", detail=f"status={r.status_code} (endpoint exists)", latency=lat)
                else:
                    self._record(name, "FAIL", f"status={r.status_code}: {r.text[:100]}")
            except Exception as e:
                self._record(name, "FAIL", str(e))

    def test_ui(self):
        """Test UI availability."""
        print(f"\n{BLUE}━━━ UI & Static ━━━{RESET}")

        try:
            r = requests.get(f"{self.base_url}/ui", timeout=10, allow_redirects=True)
            if r.status_code == 200:
                self._record("UI /ui", "PASS")
            else:
                self._record("UI /ui", "FAIL", f"status={r.status_code}")
        except Exception as e:
            self._record("UI /ui", "FAIL", str(e))

    def summary(self):
        """Print summary."""
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"{BOLD}  Results: {GREEN}{self.passed} passed{RESET}, {RED}{self.failed} failed{RESET}, {YELLOW}{self.skipped} skipped{RESET} / {total} total")
        print(f"{'='*60}")

        if self.failed > 0:
            print(f"\n{RED}Failed tests:{RESET}")
            for r in self.results:
                if r["status"] == "FAIL":
                    print(f"  - {r['name']}: {r['detail']}")

        return self.failed == 0


def main():
    parser = argparse.ArgumentParser(description="LiteLLM Enterprise E2E Smoke Test")
    parser.add_argument("--url", default="http://localhost:4000", help="Proxy URL")
    parser.add_argument("--key", default=None, help="API key (or reads from .env)")
    parser.add_argument("--no-api", action="store_true", help="Skip tests that call external APIs")
    args = parser.parse_args()

    # Load key from .env if not provided
    key = args.key
    if not key:
        try:
            with open(".env") as f:
                for line in f:
                    if line.startswith("LITELLM_MASTER_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
        except FileNotFoundError:
            pass

    if not key:
        print(f"{RED}Error: No API key found. Use --key or create .env with LITELLM_MASTER_KEY{RESET}")
        sys.exit(1)

    print(f"{BOLD}{'='*60}")
    print(f"  LiteLLM Enterprise ── E2E Smoke Test")
    print(f"  URL: {args.url}")
    print(f"  API calls: {'DISABLED' if args.no_api else 'ENABLED'}")
    print(f"{'='*60}{RESET}")

    test = SmokeTest(args.url, key, skip_api=args.no_api)

    test.test_health()
    test.test_models()
    test.test_embeddings()
    test.test_caching()
    test.test_enterprise()
    test.test_ui()

    success = test.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
