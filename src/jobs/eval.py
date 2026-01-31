# Script to generate eval commands for specific datasets.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--training-path", "-p", type=str, required=True)
