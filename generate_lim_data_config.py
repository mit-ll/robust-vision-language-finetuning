'''
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research
and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.
'''

# Script for defining limited data experiments within lim_data_config.json

import json

ARCHS = [
    "vitb32_pretrained_finetune",
    "vitb32_pretrained_linearprob",
    "vitb32_clip_finetune",
    "vitb32_clip_linearprob",
]

PERCS = [
    0,
    .03,
    .05,
    .1,
    .3,
    .5,
    .7,
    .9,
    1,
]

LRS = [
    1e-2,
    1e-3,
    1e-4,
    1e-5,
]

params = {
    'archs': ARCHS,
    'percs': PERCS,
    'lrs': LRS,
}

with open('lim_data_config.json', 'w') as f:
    json.dump(params, f)