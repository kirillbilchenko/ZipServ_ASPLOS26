# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Create a private off/on trial from the stopped CT container, without its secret mount."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--mode', choices=('off', 'on'), required=True)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--run', required=True, help='Unique alphanumeric trial label')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--dispatch', choices=('fixed', 'tokens', 'shape'), default='tokens')
    parser.add_argument('--partial-dtype', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--large-batch-tokens', type=int, default=768)
    parser.add_argument('--workspace-bytes', type=int, default=64 * 2**20)
    parser.add_argument('--decode-workspace-bytes', type=int, default=256 * 2**20)
    args = parser.parse_args()
    root = args.workdir.resolve()
    private_root = (str(root).startswith('/tmp/zipserv-ct-extra.') or
                    (root.name.startswith('zipserv-ct-extra-') and root.parent.name == 'experiments'))
    if (not private_root or not args.run.isalnum() or args.large_batch_tokens < 1
            or min(args.workspace_bytes, args.decode_workspace_bytes) < 0):
        parser.error('Require a private zipserv-ct-extra workspace, alphanumeric run label and valid budgets')
    template = f'mxwave-glm53-ct-live-20261022-r{args.rank}'
    original = json.loads(subprocess.check_output(['docker', 'inspect', template]))[0]
    if original['State']['Running'] or original['Image'] != 'sha256:39a03677234ea74e35544d301bd14716c1ae9695ad8d91b16e2c76545e26e8d2':
        raise RuntimeError('Require the stopped, qualified CT template')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if active.strip():
        raise RuntimeError('GPU is occupied; do not disturb another workload')
    env = {}
    for value in original['Config']['Env']:
        key, _, content = value.partition('=')
        if (any(word in key.upper() for word in ('SECRET', 'PASSWORD', 'API_KEY', 'ACCESS_KEY', 'AUTHORIZATION'))
                or key.upper().endswith(('_TOKEN', '_TOKEN_FILE'))):
            continue
        env[key] = content
    if env.get('MXWAVE_COMPARE_ARM') != 'B' or env.get('MXWAVE_MOE_MODE') != 'aligned':
        raise RuntimeError('Both trial arms must retain the existing fused CT recipe')
    run_root = root / (args.run + '-' + args.mode)
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.mkdir()
    (run_root / 'reports').mkdir()
    bootstrap = root / 'bootstrap' / 'zipserv_ct_extra_trial-0.0.0.dist-info'
    bootstrap.mkdir(parents=True, exist_ok=True)
    (bootstrap / 'METADATA').write_text('Metadata-Version: 2.1\nName: zipserv-ct-extra-trial\nVersion: 0.0.0\n')
    (bootstrap / 'entry_points.txt').write_text('[vllm.general_plugins]\nzipserv_ct_extra = ct_bf16_extra_plugin:install\n')
    env.update(ZIPSERV_CT_EXTRA_BF16=args.mode, ZIPSERV_LIBRARY='/zipserv-extra/build/libzipserv_vllm.so',
               ZIPSERV_CT_BF16_DISPATCH=args.dispatch, ZIPSERV_CT_BF16_PARTIAL_DTYPE=args.partial_dtype,
               ZIPSERV_CT_BF16_LARGE_BATCH_TOKENS=str(args.large_batch_tokens),
               ZIPSERV_CT_BF16_WORKSPACE_BYTES=str(args.workspace_bytes),
               ZIPSERV_CT_BF16_DECODE_WORKSPACE_BYTES=str(args.decode_workspace_bytes),
               PYTHONDONTWRITEBYTECODE='1',
               ZIPSERV_CT_SOURCE='/model', ZIPSERV_CT_REPORTS='/reports', VLLM_SERVER_DEV_MODE='1',
               PYTHONPATH=env.get('PYTHONPATH', '') + ':/zipserv-extra/bootstrap:/zipserv-extra/tools:/zipserv-extra/integrations/vllm')
    command = list(original['Config']['Cmd'])
    for flag, value in (('--port', '18795'), ('--master-port', '29766'),
                        ('--host', '127.0.0.1'),
                        ('--worker-extension-cls', 'ct_bf16_extra_plugin.CTExtraWorker')):
        if flag in command:
            command[command.index(flag) + 1] = value
        elif flag == '--worker-extension-cls':
            raise RuntimeError('The original CT worker extension is missing')
    if any(flag.startswith('--api-key') for flag in command):
        raise RuntimeError('Unexpected inline API authentication; refuse to copy it')
    mounts = []
    cache_source = None
    for mount in original['Mounts']:
        destination = mount['Destination']
        if destination.startswith('/run/secrets'):
            continue
        source = mount['Source']
        if destination == '/cache':
            cache_source = source
            source = str(run_root / 'cache')
        elif destination == '/reports':
            source = str(run_root / 'reports')
        elif mount['RW']:
            raise RuntimeError('Unexpected writable production mount: ' + destination)
        mounts.append({'source': source, 'destination': destination, 'rw': mount['RW']})
    if cache_source is None:
        raise RuntimeError('CT cache mount is missing')
    # Keep each arm's generated files private. Seed from the same stopped baseline.
    (run_root / 'cache').mkdir()
    if not args.preflight:
        # The existing cache contains root-owned CUDA files from the CT container.
        # Read it through a read-only mount; write only the private trial directory.
        subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--cpus', '2', '--memory', '2g',
                        '-v', cache_source + ':/seed:ro',
                        '-v', str(run_root / 'cache') + ':/destination:rw',
                        '--entrypoint', 'timeout', original['Image'], '300',
                        'cp', '-a', '--reflink=auto', '/seed/.', '/destination'], check=True)
    mounts.append({'source': str(root), 'destination': '/zipserv-extra', 'rw': False})
    name = f'zipserv-ct-extra-{args.run}-{args.mode}-r{args.rank}'
    launch = ['docker', 'run', '--name', name, '--gpus', 'all', '--network', 'host', '--ipc', 'host']
    host = original['HostConfig']
    for device in host.get('Devices') or []:
        launch += ['--device', f'{device["PathOnHost"]}:{device["PathInContainer"]}:{device["CgroupPermissions"]}']
    for capability in host.get('CapAdd') or []:
        launch += ['--cap-add', capability]
    for option in host.get('SecurityOpt') or []:
        launch += ['--security-opt', option]
    for limit in original['HostConfig'].get('Ulimits') or []:
        if limit['Name'] == 'core':
            continue
        launch += ['--ulimit', f'{limit["Name"]}={limit["Soft"]}:{limit["Hard"]}']
    launch += ['--ulimit', 'core=0']
    for key, value in env.items():
        launch += ['-e', key + '=' + value]
    for mount in mounts:
        launch += ['-v', f'{mount["source"]}:{mount["destination"]}:{"rw" if mount["rw"] else "ro"}']
    if args.preflight:
        launch += ['--cpus', '4', '--memory', '8g', '--entrypoint', 'timeout', original['Image'],
                   '120', 'python3', '-c',
                   'import ct_bf16_extra_plugin as p; p.install(); p.install(); print("CT extra adapter preflight passed")']
    else:
        launch += ['-d', '--entrypoint', 'timeout', original['Image'], '3600', 'vllm', *command]
    receipt = {'template': template, 'name': name, 'mode': args.mode, 'rank': args.rank,
               'image': original['Image'], 'cmd': command, 'env': env, 'mounts': mounts,
               'host_config': {key: host.get(key) for key in
                               ('NetworkMode', 'IpcMode', 'Devices', 'CapAdd', 'SecurityOpt', 'Ulimits')},
               'preflight': args.preflight,
               'sources': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((root / 'integrations/vllm/zipserv_vllm').glob('*.py'))},
               'library_sha256': hashlib.sha256((root / 'build/libzipserv_vllm.so').read_bytes()).hexdigest(),
               'tools': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((root / 'tools').glob('*.py'))},
               'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (run_root / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    process = subprocess.run(launch, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    receipt['returncode'] = process.returncode
    (run_root / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (run_root / 'launch.log').write_text(process.stdout)
    print(json.dumps({'name': name, 'returncode': process.returncode, 'run_root': str(run_root),
                      'output_tail': process.stdout[-2000:]}), flush=True)
    process.check_returncode()


if __name__ == '__main__':
    main()
