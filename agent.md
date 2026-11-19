# Agent instructions

After every feature or code change, validate with a tiny CPU model (~0.1M params), not the 100M model.

Run:

- `.venv/bin/python -m py_compile model/*.py pretrain/*.py code_test/*.py`
- `.venv/bin/python code_test/test_train_tiny.py`
- `.venv/bin/python code_test/test_kv_cache.py`
- `.venv/bin/python code_test/test_data_pipeline.py`
- `.venv/bin/python code_test/test_distill_tiny.py`
- `.venv/bin/python code_test/test_sft_tiny.py`
- `.venv/bin/python code_test/test_train_curves_real.py`

Never read, print, or commit `.env`; keep `HF_TOKEN` in env/Kaggle Secrets. Update READMEs/runbook when behavior or config changes. Commit and push only after tests pass.
