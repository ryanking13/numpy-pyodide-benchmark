#!/bin/bash

set -e

which node
node --version
npm install
node benchmark.mjs
