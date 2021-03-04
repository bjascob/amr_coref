#!/usr/bin/python3
import os
import json
from   shutil import copyfile


meta = {
    "model_type":"coref",
    "version":"0.1.0",
    "date":"2021-03-03",
    "inference_module":"coref.inference",
    "inference_class":"Inference",
    "model_fn":"amr_coref.pt",
    "kwargs":{}
}


if __name__ == '__main__':
    model_in_dir  = 'data/model'
    model_out_dir = 'data/model_coref-v%s' % meta['version']

    # Create the directory and copy a copy of files
    print('Copying model to', model_out_dir)
    os.makedirs(model_out_dir)
    copyfile(os.path.join(model_in_dir, 'amr_coref.pt'), os.path.join(model_out_dir, 'amr_coref.pt'))
    copyfile(os.path.join(model_in_dir, 'config.json'),   os.path.join(model_out_dir, 'config.json'))

    # Write the metadata
    with open(os.path.join(model_out_dir, 'amrlib_meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
