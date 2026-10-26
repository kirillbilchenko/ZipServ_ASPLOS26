# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Build pinned raw and exact five-bit NVFP4 scale consumers with standard math."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arch', choices=('80', '86', '89', '90', '100', '120', '121'), required=True)
    parser.add_argument('--stage-only', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=('raw', 'delta5'), default=['raw', 'delta5'])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory to preserve previous build identities')
    package = Path(__file__).resolve().parent
    vendor = package / 'native/nvfp4'
    spec = importlib.util.spec_from_file_location('zipserv_nvfp4_stager', vendor / 'stage.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.stage(args.source.resolve(), args.output.resolve())
    staged = args.output.resolve() / 'staged'
    manifest = {'format': 'zipserv-nvfp4-native-v1', 'abi_version': 1, 'features': 7,
                'architecture': 'SM' + args.arch, 'representation': 'exact-byte-offset-5',
                'scale_layout': 'post-Marlin S0E5M3 bytes', 'group_size': 16,
                'math': 'standard; fast-math forbidden', 'shared_reuse_barrier': True,
                'upstream': original['source'], 'builder_sha256': digest(Path(__file__)),
                'vendor_sha256': {p.name: digest(p) for p in vendor.iterdir() if p.is_file()},
                'source_sha256': {str(p.relative_to(staged)): digest(p) for p in staged.rglob('*') if p.is_file()},
                'libraries': {}}
    if not args.stage_only:
        import torch
        include = Path(torch.__file__).parent / 'include'
        cuda_path = os.environ.get('CUDA_PATH')
        compiler = str(Path(cuda_path) / 'bin/nvcc') if cuda_path else shutil.which('nvcc')
        if compiler is None:
            raise RuntimeError('CUDA nvcc is required')
        manifest['compiler'] = subprocess.check_output([compiler, '--version'], text=True)
        for variant in dict.fromkeys(args.variants):
            library = args.output.resolve() / ('libzipserv_nvfp4_' + variant + '.so')
            command = [compiler, '-std=c++17', '-O3', '-arch=sm_' + args.arch, '--shared',
                       '-Xcompiler=-fPIC', '--expt-extended-lambda', '--expt-relaxed-constexpr',
                       '-lineinfo', '-Xptxas=-v', '-I' + str(staged),
                       '-I' + str(staged / 'libtorch_stable/moe/marlin_moe_wna16'), '-I' + str(include),
                       '-DNV_COMPRESSED=' + ('0' if variant == 'raw' else '3'),
                       '-DMARLIN_NAMESPACE_NAME=zipserv_nvfp4_' + variant,
                       str(staged / 'host.cu'), '-o', str(library)]
            with (args.output / ('build-' + variant + '.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            manifest['libraries'][variant] = {'path': library.name, 'sha256': digest(library), 'command': command}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'staged' if args.stage_only else 'built', 'output': str(args.output)}))


if __name__ == '__main__':
    main()
