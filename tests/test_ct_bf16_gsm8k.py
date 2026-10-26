# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""The smaller CT experiment must retain a frozen, matched evaluation contract."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import eval_ct_bf16_gsm8k as evaluation


def test_subset_is_frozen_unique_and_unchanged_between_arms(monkeypatch):
    parent = {'cases': [{'source_index': i, 'prompt': f'problem {i}', 'expected': str(i)}
                        for i in range(1319)], 'limits': [], 'max_tokens': 1024,
              'temperature': 0, 'seed': 42, 'concurrency': 4}
    raw = json.dumps(parent).encode()
    monkeypatch.setattr(evaluation, 'PARENT_SHA', evaluation.sha(raw))
    a = evaluation.freeze(raw); b = evaluation.freeze(raw)
    assert a == b
    assert len(a['cases']) == len(set(a['selection']['indices'])) == 300
    assert [c['source_index'] for c in a['cases']] == a['selection']['indices']
    assert a['selection']['indices'] != list(range(300))
    for case in a['cases']:assert case == parent['cases'][case['source_index']]
    assert a['max_tokens'] == 1024 and a['temperature'] == 0 and a['concurrency'] == 4
    with pytest.raises(ValueError, match='hash-pinned'):
        evaluation.freeze(raw+b' ')


def test_launch_matching_allows_only_the_extra_bf16_toggle_and_private_paths():
    a = {key: 'same' for key in ('image','cmd','sources','tools','library_sha256','host_config')}
    a['env'] = {'ZIPSERV_CT_EXTRA_BF16': 'off', 'GLM_LV_DRAFT_FP8_KV': '1'}
    a['mounts'] = [{'source':'/off/cache','destination':'/cache','rw':True},
                   {'source':'/off/reports','destination':'/reports','rw':True},
                   {'source':'/frozen/model','destination':'/model','rw':False}]
    b = copy.deepcopy(a); b['env']['ZIPSERV_CT_EXTRA_BF16']='on'
    b['mounts'][0]['source']='/on/cache';b['mounts'][1]['source']='/on/reports'
    assert evaluation.normalized_launch(a) == evaluation.normalized_launch(b)
    reordered = copy.deepcopy(b); reordered['mounts'].reverse()
    assert evaluation.normalized_launch(a) == evaluation.normalized_launch(reordered)
    reordered['mounts'][0]['rw'] = True
    assert evaluation.normalized_launch(a) != evaluation.normalized_launch(reordered)
    reordered['mounts'].append(copy.deepcopy(reordered['mounts'][0]))
    with pytest.raises(ValueError, match='Duplicate mount destinations'):
        evaluation.normalized_launch(reordered)
    b['env']['GLM_LV_DRAFT_FP8_KV']='0'
    assert evaluation.normalized_launch(a) != evaluation.normalized_launch(b)
    b['env']['GLM_LV_DRAFT_FP8_KV']='1';b['mounts'][2]['source']='/different/model'
    assert evaluation.normalized_launch(a) != evaluation.normalized_launch(b)
