#!/bin/bash
g++ transform.cpp -o transform `pkg-config opencv --cflags --libs`
./transform
