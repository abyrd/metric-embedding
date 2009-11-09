#!/bin/bash

# lossy mpg4
rm anim/vel.mp4
rm anim/err.mp4

# launch in background to use 2 processors for encoding
ffmpeg -qscale 1 -r $1 -i img/vel%03d.png anim/vel.mp4
ffmpeg -qscale 1 -r $1 -i img/err%03d.png anim/err.mp4
ffmpeg -qscale 1 -r $1 -i img/debug%03d.png anim/debug.mp4

