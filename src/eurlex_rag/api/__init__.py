"""The HTTP service (needs the ``serve`` extra: ``pip install -e ".[serve]"``).

Nothing is imported here on purpose: ``import eurlex_rag`` must keep working
without FastAPI installed. Use ``eurlex_rag.api.app.create_app``.
"""
