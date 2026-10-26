# Source attribution

The optional vLLM MXFP4 expert-scale integration adapts Apache-2.0 code from
[MxWave](https://github.com/kirillbilchenko/mxstream). Exact original native and
Python source hashes are recorded in
`integrations/vllm/zipserv_vllm/mxfp4-provenance.json`. Original notices and the
license are retained with the native source. ZipServ's changes add its own
namespace, bundled encoder/runtime, explicit activation, conservative fallback,
ownership reporting and a shared-memory reuse barrier.

The native build stages hash-pinned vLLM grouped-Marlin sources under Apache-2.0.
Its exact upstream revision and source hashes are in
`integrations/vllm/native/mxfp4/moe_sources.json`; the build retains the upstream
license and notices. Do not remove these attributions when redistributing a
source or native build. Further adapted code should record its origin, license,
exact source identity and the scope of the changes.

`tools/eval_ct_gsm8k.py` adapts MxWave's Apache-2.0 GSM8K protocol and numeric
scoring from `integrations/glm53_mxfp4/gsm8k_benchmark.py` (SHA256
`9a286af2ba19d9f818d300631a17c08939de483eea874b118f16c9e1a6eec79a`)
and `numeric_scoring.py` (SHA256
`1b6d896cc4ad46085bd5ec836fabe3201a0b27661d57a921f6ccc459a6c329d2`).
Changes add a standalone durable client, complete-run validation, explicit
protocol identities and paired answer outcomes. Historical reference KL results
are intentionally absent from the new experiment's protocol.

The optional NVFP4 scale consumer adapts MxWave's Apache-2.0
`integrations/glm53_nvfp4` components. Original file identities and the scope
of modifications are recorded in
`integrations/vllm/zipserv_vllm/nvfp4-provenance.json`. Keep this manifest,
the package NOTICE and the native LICENSE when redistributing the adapter.

# CPU validation

Use Python 3.12 or newer and a C++17 compiler. Install a suitable Torch wheel
for the host first if an existing GPU environment must be preserved, then:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check integrations/vllm/zipserv_vllm
make -C tests cpu
python -m build --wheel --no-isolation --outdir /tmp/zipserv-wheel integrations/vllm
```

The default pytest discovery is limited to `tests/`; archived experiment
snapshots are not a second test suite. Two zstd-dependent tests skip when the
system shared library is unavailable. Metadata-only tests still run.
CPU CI exercises these checks without vLLM or CUDA. GPU kernels and serving
changes additionally need the relevant device checks and matched serving
measurements described in `integrations/vllm/README.md`.
