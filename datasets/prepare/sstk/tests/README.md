# SSTK Prepare Tests

Tests for `prepare.py` (images → MDS).

## Running

From project root:

```bash
python -m pytest datasets/prepare/sstk/tests/ -v -m "not integration"
python -m pytest datasets/prepare/sstk/tests/test_prepare.py -v
```

Integration (10 images → MDS → read → save PNGs):

```bash
python -m pytest datasets/prepare/sstk/tests/test_prepare.py::TestIntegrationPrepareAndRead -v
```
