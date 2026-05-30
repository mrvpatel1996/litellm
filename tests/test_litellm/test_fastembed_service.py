import asyncio
import sys
import types

import pytest


def _install_fake_fastembed(monkeypatch):
    """Install a fake `fastembed` module so tests don't download ONNX models."""
    mod = types.ModuleType("fastembed")

    class FakeDense:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name

        def embed(self, texts, **kwargs):
            for i, _ in enumerate(texts):
                yield [float(i)] * 384

    class FakeSparseEmb:
        def __init__(self, indices, values):
            self.indices = indices
            self.values = values

    class FakeSparse:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name

        def embed(self, texts, **kwargs):
            for i, _ in enumerate(texts):
                yield FakeSparseEmb([i, i + 1], [0.5, 0.25])

    mod.TextEmbedding = FakeDense
    mod.SparseTextEmbedding = FakeSparse
    monkeypatch.setitem(sys.modules, "fastembed", mod)


def test_embed_dense_shape(monkeypatch):
    _install_fake_fastembed(monkeypatch)
    from litellm.llms.fastembed.embed import FastEmbedService

    svc = FastEmbedService()
    vecs = svc.embed_dense(["hello", "world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384


def test_embed_sparse_shape(monkeypatch):
    _install_fake_fastembed(monkeypatch)
    from litellm.llms.fastembed.embed import FastEmbedService

    svc = FastEmbedService()
    sparse = svc.embed_sparse(["hello"])
    assert sparse[0]["indices"] == [0, 1]
    assert sparse[0]["values"] == [0.5, 0.25]


def test_singleton_accessor(monkeypatch):
    _install_fake_fastembed(monkeypatch)
    import litellm.llms.fastembed.embed as embed_mod

    embed_mod._service_singleton = None
    a = embed_mod.get_fastembed_service()
    b = embed_mod.get_fastembed_service()
    assert a is b


def test_async_dense(monkeypatch):
    _install_fake_fastembed(monkeypatch)
    from litellm.llms.fastembed.embed import FastEmbedService

    svc = FastEmbedService()
    vecs = asyncio.run(svc.aembed_dense(["x"]))
    assert len(vecs) == 1 and len(vecs[0]) == 384
