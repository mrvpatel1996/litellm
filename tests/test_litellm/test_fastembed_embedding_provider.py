import sys
import types

import pytest


def _install_fake_fastembed(monkeypatch):
    mod = types.ModuleType("fastembed")

    class FakeDense:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name

        def embed(self, texts, **kwargs):
            for _ in texts:
                yield [0.1] * 384

    class FakeSparse:
        def __init__(self, model_name, **kwargs):
            pass

        def embed(self, texts, **kwargs):
            for _ in texts:
                yield types.SimpleNamespace(indices=[0], values=[1.0])

    mod.TextEmbedding = FakeDense
    mod.SparseTextEmbedding = FakeSparse
    monkeypatch.setitem(sys.modules, "fastembed", mod)
    import litellm.llms.fastembed.embed as embed_mod

    embed_mod._service_singleton = None


def test_provider_resolves():
    from litellm import get_llm_provider

    _model, provider, _key, _base = get_llm_provider(
        model="fastembed/BAAI/bge-small-en-v1.5"
    )
    assert provider == "fastembed"


def test_embedding_returns_openai_shape(monkeypatch):
    _install_fake_fastembed(monkeypatch)

    import litellm

    resp = litellm.embedding(model="fastembed/BAAI/bge-small-en-v1.5", input=["a", "b"])
    assert len(resp.data) == 2
    assert len(resp.data[0]["embedding"]) == 384
    assert resp.data[0]["index"] == 0
