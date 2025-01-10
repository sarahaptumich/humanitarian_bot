#!/bin/bash

# Update and install dependencies
sudo apt-get update
sudo apt-get install -y build-essential libsqlite3-dev

# Download and build the latest SQLite version
wget https://www.sqlite.org/2025/sqlite-autoconf-3430200.tar.gz
tar -xvf sqlite-autoconf-3430200.tar.gz
cd sqlite-autoconf-3430200
./configure
make
sudo make install

# Verify SQLite version
sqlite3 --version
