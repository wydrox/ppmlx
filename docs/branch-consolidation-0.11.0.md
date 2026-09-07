# Branch consolidation for 0.11.0

Date: 2026-09-07

This record describes the source review for the 0.11.0 release candidate. It does not claim that all branches are merged or that final checks are complete.

The review compared branch patches with the current `main` tree. Exact-current branch files were retained from `main`. Superseded contract, provider, normalization, profile, security, Homebrew, and release histories were also retained from `main` because later commits contain the current forms.

The retained experimental source includes the local agent, voice support, prompt cache, draft decoding, RAG and document processing, templates, gateway, batch processing, API documentation, and automatic model routing. The policy router remains in `ppmlx/router.py`. Optional `model:auto` selection uses `ppmlx/auto_router.py`.

The extraction claim race fix remains a targeted integration item. The normalization gate branch adds regression tests for commit-bound fixture evidence. The older `origin/dev` branch needs selective review for its Gemma 4 streaming and safe error changes. These items require the final root integration and test pass.

The strict proxy Agent IR runtime keeps tool execution outside ppmlx. The explicit `ppmlx agent` command is the local path that can execute tools.
