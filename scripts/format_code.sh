#!/bin/bash

# exit when any command fails
set -e

if [[ ! -f setup.cfg ]] || [[ ! -d arabic_llm_benchmark ]] || [[ ! -d assets ]] || [[ ! -d scripts ]] || [[ ! -d tests ]]
then
  echo "format_code.sh must be run from the root of the repository"
  exit 1
fi

ufmt format assets
ufmt format arabic_llm_benchmark
ufmt format tests
