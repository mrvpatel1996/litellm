#!/usr/bin/env python3
"""
LiteLLM Cache Monitor — shows cache hit rates and cost savings.

Usage:
  python3 cache_monitor.py              # Show stats
  python3 cache_monitor.py --watch      # Continuous monitoring  
  python3 cache_monitor.py --clear      # Clear cache collection
"""
import argparse
import json
import sys
import requests

QDRANT_URL = "http://localhost:6333"
CACHE_COLLECTION = "litellm_cache"
LITELLM_URL = "http://localhost:4000"


def get_cache_stats():
    """Get cache collection stats from Qdrant."""
    r = requests.get(f"{QDRANT_URL}/collections/{CACHE_COLLECTION}")
    if r.status_code != 200:
        return None
    return r.json()["result"]


def get_cached_prompts(limit=10):
    """Get recent cached prompts from Qdrant."""
    r = requests.post(
        f"{QDRANT_URL}/collections/{CACHE_COLLECTION}/points/scroll",
        json={
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        },
        headers={"Content-Type": "application/json"},
    )
    if r.status_code != 200:
        return []
    return r.json()["result"]["points"]


def clear_cache():
    """Delete the entire cache collection."""
    r = requests.delete(f"{QDRANT_URL}/collections/{CACHE_COLLECTION}")
    if r.status_code == 200:
        print(f"✅ Cache collection '{CACHE_COLLECTION}' deleted")
        print("   It will be recreated on next cache miss")
    else:
        print(f"❌ Failed: {r.text}")


def show_stats():
    """Display cache statistics."""
    stats = get_cache_stats()
    if stats is None:
        print(f"❌ Cache collection '{CACHE_COLLECTION}' not found")
        print("   Cache will be created automatically on first request")
        return

    print("╔══════════════════════════════════════════════╗")
    print("║     LiteLLM Semantic Cache Dashboard         ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  Collection:   {CACHE_COLLECTION:<30}║")
    print(f"║  Status:       {stats['status']:<30}║")
    print(f"║  Cached items: {stats['points_count']:<30}║")
    print(f"║  Vector dim:   {stats['config']['params']['vectors']['size']:<30}║")
    print(f"║  Distance:     {stats['config']['params']['vectors']['distance']:<30}║")
    print(f"║  Quantization: {list(stats['config']['quantization_config'].keys())[0]:<30}║")
    print("╠══════════════════════════════════════════════╣")
    print("║  Recent cached prompts:                      ║")

    points = get_cached_prompts(limit=5)
    if not points:
        print("║  (no cached prompts yet)                     ║")
    for i, p in enumerate(points):
        text = p["payload"].get("text", "")[:35]
        has_response = "response" in p["payload"]
        print(f"║  {i+1}. {text:<38}{'✅' if has_response else '⏳'}║")

    print("╠══════════════════════════════════════════════╣")
    print("║  Cost savings estimate:                      ║")
    cached = stats["points_count"]
    # Assume gpt-4o ~$5/1M tokens avg, ~500 tokens per cached response
    est_savings = cached * 500 * 5.0 / 1_000_000
    print(f"║  Cached responses: {cached:<26}║")
    print(f"║  Est. tokens saved: ~{cached * 500:<24}║")
    print(f"║  Est. cost saved:   ~${est_savings:.4f}{' ' * 23}║")
    print("╚══════════════════════════════════════════════╝")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiteLLM Cache Monitor")
    parser.add_argument("--clear", action="store_true", help="Clear cache")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring")
    args = parser.parse_args()

    if args.clear:
        clear_cache()
    elif args.watch:
        import time
        while True:
            print("\033[2J\033[H")  # Clear screen
            show_stats()
            time.sleep(5)
    else:
        show_stats()
