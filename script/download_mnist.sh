#!/bin/sh

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

BASE_URL="https://raw.githubusercontent.com/fgnt/mnist/master"

FILES="
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
"

echo "Downloading MNIST dataset..."

for file in $FILES
do
    if [ -f "$DATA_DIR/${file%.gz}" ]; then
        echo "$file already exists, skipping."
        continue
    fi

    echo "Downloading $file..."

    if command -v curl >/dev/null 2>&1; then
        curl -L "$BASE_URL/$file" -o "$DATA_DIR/$file"
    elif command -v wget >/dev/null 2>&1; then
        wget "$BASE_URL/$file" -O "$DATA_DIR/$file"
    else
        echo "Error: curl or wget is required."
        exit 1
    fi

    gzip -d "$DATA_DIR/$file"
done

echo "MNIST downloaded successfully!"