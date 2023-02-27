# Copyright (c) OpenMMLab. All rights reserved.

import argparse

from mmdeploy import __version__
from mmdeploy.commands import COMMAND_REGISTRY


def main():
    parser = argparse.ArgumentParser(description='Run Easy Object Detector')
    parser.add_argument('--version', action='version', version=__version__)
    subparsers = parser.add_subparsers(title='subcommands')

    for subname, subcommand in COMMAND_REGISTRY._module_dict.items():
        subcommand().add_subparser(subname, subparsers)

    args = parser.parse_args()

    if 'run' in dir(args):
        args.run(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
