#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is modified by Ze Yang.

"""
 Command line tool to download various preprocessed data sources & checkpoints
"""

import gzip
import os
import pathlib

import argparse
import wget


RESOURCES_MAP = {
    'models.dialogpt-medium':{
        's3_url': 'https://convaisharables.blob.core.windows.net/lsp/multiref/medium_ft.pkl',
        'original_ext': '.pkl',
        'compressed': False,
        'desc': 'Weights for the released medium version of DialoGPT.'
    },
    'models.gpt2-small':{
        's3_url': 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin',
        'original_ext': '.bin',
        'compressed': False,
        'desc': 'Weights for the gpt-2 (small).'

    },
}


def unpack(gzip_file: str, out_file: str):
    print('Uncompressing ', gzip_file)
    input = gzip.GzipFile(gzip_file, 'rb')
    s = input.read()
    input.close()
    output = open(out_file, 'wb')
    output.write(s)
    output.close()
    print('Saved to ', out_file)


def download_resource(s3_url: str, original_ext: str, compressed: bool, resource_key: str, out_dir: str) -> str:
    print('Loading from ', s3_url)

    # create local dir
    path_names = resource_key.split('.')

    root_dir = out_dir if out_dir else './'
    save_root = os.path.join(root_dir, *path_names[:-1])  # last segment is for file name

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file = os.path.join(save_root, path_names[-1] + ('.tmp' if compressed else original_ext))

    if os.path.exists(local_file):
        print('File already exist ', local_file)
        return save_root

    wget.download(s3_url, out=local_file)

    print('Saved to ', local_file)

    if compressed:
        uncompressed_file = os.path.join(save_root, path_names[-1] + original_ext)
        unpack(local_file, uncompressed_file)
        os.remove(local_file)
    return save_root


def download_file(s3_url: str, out_dir: str, file_name: str):
    print('Loading from ', s3_url)
    local_file = os.path.join(out_dir, file_name)

    if os.path.exists(local_file):
        print('File already exist ', local_file)
        return

    wget.download(s3_url, out=local_file)
    print('Saved to ', local_file)


def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            print('no resources found for specified key')
        return
    download_info = RESOURCES_MAP[resource_key]

    s3_url = download_info['s3_url']

    save_root_dir = None
    if isinstance(s3_url, list):
        for i, url in enumerate(s3_url):
            save_root_dir = download_resource(url,
                                              download_info['original_ext'],
                                              download_info['compressed'],
                                              '{}_{}'.format(resource_key, i),
                                              out_dir)
    else:
        save_root_dir = download_resource(s3_url,
                                          download_info['original_ext'],
                                          download_info['compressed'],
                                          resource_key,
                                          out_dir)

    license_files = download_info.get('license_files', None)
    if not license_files:
        return

    download_file(license_files[0], save_root_dir, 'LICENSE')
    download_file(license_files[1], save_root_dir, 'README')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="./", type=str,
                        help="The output directory to download file")
    parser.add_argument("--resource", type=str,
                        help="Resource name. See RESOURCES_MAP for all possible values")
    args = parser.parse_args()
    if args.resource:
        download(args.resource, args.output_dir)
    else:
        print('Please specify resource value. Possible options are:')
        for k, v in RESOURCES_MAP.items():
            print('Resource key={}  description: {}'.format(k, v['desc']))


if __name__ == '__main__':
    main()