#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"
cd $DIR

FILE=models.tar.gz
URL=https://people.eecs.berkeley.edu/~pathak/noreward-rl/resources/$FILE
CHECKSUM=26bdf54e9562e23750ebc2ef503204b1

if [ ! -f $FILE ]; then
  echo "Downloading the curiosity-driven RL trained models (6MB)..."
  wget $URL -O $FILE
  echo "Unzipping..."
  tar zxvf $FILE
  mv models/* .
  rm -rf models
  echo "Downloading Done."
else
  echo "File already exists. Checking md5..."
fi

os=`uname -s`
if [ "$os" = "Linux" ]; then
  checksum=`md5sum $FILE | awk '{ print $1 }'`
elif [ "$os" = "Darwin" ]; then
  checksum=`cat $FILE | md5`
elif [ "$os" = "SunOS" ]; then
  checksum=`digest -a md5 -v $FILE | awk '{ print $4 }'`
fi
if [ "$checksum" = "$CHECKSUM" ]; then
  echo "Checksum is correct. File was correctly downloaded."
  exit 0
else
  echo "Checksum is incorrect. DELETE and download again."
fi
