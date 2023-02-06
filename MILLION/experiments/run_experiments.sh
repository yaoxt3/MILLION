#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "script dir is " $DIR

python3 ml10_language_instructions.py --log_dir=$DIR/../logs/ml10/language_instructions

