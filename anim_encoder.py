#!/usr/bin/env python3
# Copyright (c) 2012, Sublime HQ Pty Ltd
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import json
import shlex
import shutil
import hashlib
from time import time
from argparse import ArgumentParser
from subprocess import DEVNULL, STDOUT, check_call

import cv2
import scipy.ndimage as me
import imageio.v2 as imageio
from numpy import *

# How many pixels can be wasted in the name of combining neighbouring changed
# regions.
SIMPLIFICATION_TOLERANCE = 512

# Maximum height of the image.
MAX_PACKED_HEIGHT = 20000


def parse_args():
    """
    Parses command line arguments.
    """
    parser = ArgumentParser(
        prog="anim_encoder", description="Generate packed PNG of an animation"
    )

    parser.add_argument(
        "animation_path",
        nargs="+",
        help=(
            "The path(s) containing the frames for the animation. "
            "If a path points to a directory, its files will be added. "
            "Otherwise, the path is added as-is. "
            "The paths will be sorted."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "The output filename of the PNG and metadata."
            "If not specified, the filename (without extension) "
            "of the first input will be used."
        ),
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "js"],
        default="js",
        help="The format of the metadata.",
    )
    parser.add_argument(
        "-r", "--fps", type=int, default=15, help="The framerate of the animation."
    )
    parser.add_argument(
        "-p",
        "--end-pause",
        type=int,
        default=4000,
        help="The pause period (in milliseconds) before repeating the animation.",
    )
    parser.add_argument(
        "-c",
        "--compress",
        action="store_true",
        help="Attempt to compress the result PNG with pngcrush.",
    )
    parser.add_argument(
        "-C",
        "--compress-pngquant",
        action="store_true",
        help="Attempt to compress the result PNG with pngquant.",
    )
    parser.add_argument(
        "-z",
        "--compress-command",
        default="pngcrush -q {0} {1}",
        help=(
            "The command used to loselessly compress the PNG."
            "The string '{0}' or '{}' refers to the input file "
            "while '{1}' refers to the output file."
        ),
    )
    return parser.parse_args()


def slice_size(a, b):
    """
    Returns the size of a 2D slice.
    """
    return (a.stop - a.start) * (b.stop - b.start)


def combine_slices(a, b, c, d):
    """
    Combines two 2D slices together.
    """
    return (
        slice(min(a.start, c.start), max(a.stop, c.stop)),
        slice(min(b.start, d.start), max(b.stop, d.stop)),
    )


def slices_intersect(a, b, c, d):
    """
    Checks whether two 2D slices intersect each other.
    """
    if a.start >= c.stop:
        return False
    if c.start >= a.stop:
        return False
    if b.start >= d.stop:
        return False
    if d.start >= b.stop:
        return False
    return True


def simplify(boxes, tol=0):
    """
    COmbine a large set of rectangles into a smaller set of rectangles,
    minimising the number of additional pixels included in the smaller
    set of rectangles.
    """
    out = []
    for a, b in boxes:
        sz1 = slice_size(a, b)
        did_combine = False
        for i in range(len(out)):
            c, d = out[i]
            cu, cv = combine_slices(a, b, c, d)
            sz2 = slice_size(c, d)
            if slices_intersect(a, b, c, d) or (slice_size(cu, cv) <= sz1 + sz2 + tol):
                out[i] = (cu, cv)
                did_combine = True
                break
        if not did_combine:
            out.append((a, b))

    if tol != 0:
        return simplify(out, 0)
    else:
        return out


def slice_tuple_size(s):
    """
    Returns the size of a 2D slice packed in a tuple.
    """
    a, b = s
    return (a.stop - a.start) * (b.stop - b.start)


class Allocator2D:
    """
    Brute force 2D allocator.
    Used to allocate space in the packed image.
    """

    def __init__(self, rows, cols):
        self.bitmap = zeros((rows, cols), dtype=uint8)
        self.available_space = zeros(rows, dtype=uint32)
        self.available_space[:] = cols
        self.num_used_rows = 0

    def allocate(self, w, h):
        """
        Allocate a rectangle with specified width and height.
        """
        bh, bw = shape(self.bitmap)

        for row in range(bh - h + 1):
            if self.available_space[row] < w:
                continue

            for col in range(bw - w + 1):
                if self.bitmap[row, col] == 0:
                    if not self.bitmap[row : row + h, col : col + w].any():
                        self.bitmap[row : row + h, col : col + w] = 1
                        self.available_space[row : row + h] -= w
                        self.num_used_rows = max(self.num_used_rows, row + h)
                        return row, col
        raise RuntimeError()


def find_matching_rect(bitmap, num_used_rows, packed, src, sx, sy, w, h):
    """
    Find matching rectangle within the packed image.
    This allows the algorithm to reuse existing rectangle.
    """
    template = src[sy : sy + h, sx : sx + w]
    bh, bw = shape(bitmap)
    image = packed[0:num_used_rows, 0:bw]

    if num_used_rows < h:
        return None

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    row, col = unravel_index(result.argmax(), result.shape)
    if (
        packed[row : row + h, col : col + w] == src[sy : sy + h, sx : sx + w]
    ).all() and (
        packed[row : row + 1, col : col + w, 0] == src[sy : sy + 1, sx : sx + w, 0]
    ).all():
        return row, col
    else:
        return None


def to_native(d):
    """
    Copy and convert numpy arrays to Python lists.
    """
    if isinstance(d, dict):
        return {k: to_native(v) for k, v in d.items()}
    if isinstance(d, list):
        return [to_native(i) for i in d]
    if type(d).__module__ == "numpy":
        return to_native(d.tolist())
    return d


def generate_animation(input_files, output, fps, end_pause, output_format):
    """
    Generate a packed PNG and metadata from a list of input files.
    """
    last_sha256 = None
    images = []
    times = []
    for t, f in enumerate(input_files):
        # Duplicate frames results in opencv terminating
        # the process with a SIGKILL during matchTemplate
        with open(f, "rb") as h:
            sha256 = hashlib.sha256(h.read()).digest()
        if sha256 == last_sha256:
            continue
        last_sha256 = sha256

        im = imageio.imread(f)
        # Remove alpha channel from image
        if im.shape[2] == 4:
            im = im[:, :, :3]
        images.append(im)
        times.append(t)

    if not images:
        print(f"No images provided in: {input_files}")
        return

    zero = images[0] - images[0]
    pairs = zip([zero] + images[:-1], images)
    diffs = [sign((b - a).max(2)) for a, b in pairs]

    # Find different objects for each frame
    img_areas = [me.find_objects(me.label(d)[0]) for d in diffs]

    # Simplify areas
    img_areas = [simplify(x, SIMPLIFICATION_TOLERANCE) for x in img_areas]

    ih, iw, _ = shape(images[0])

    # Generate a packed image
    allocator = Allocator2D(MAX_PACKED_HEIGHT, iw)
    packed = zeros((MAX_PACKED_HEIGHT, iw, 3), dtype=uint8)

    # Sort the rects to be packed by largest size first, to improve the packing
    rects_by_size = []
    for i in range(len(images)):
        src_rects = img_areas[i]

        for j in range(len(src_rects)):
            rects_by_size.append((slice_tuple_size(src_rects[j]), i, j))

    rects_by_size.sort(reverse=True)

    allocs = [[None] * len(src_rects) for src_rects in img_areas]

    print(f"Packing '{output}'")
    print(f"Number of rectangles: {len(rects_by_size)}")
    print(f"Number of frames: {len(images)}")

    t0 = time()

    for size, i, j in rects_by_size:
        src = images[i]
        src_rects = img_areas[i]

        a, b = src_rects[j]
        sx, sy = b.start, a.start
        w, h = b.stop - b.start, a.stop - a.start

        # See if the image data already exists in the packed image. This takes
        # a long time, but results in worthwhile space savings (20% in one
        # test)
        existing = find_matching_rect(
            allocator.bitmap, allocator.num_used_rows, packed, src, sx, sy, w, h
        )
        if existing:
            dy, dx = existing
            allocs[i][j] = (dy, dx)
        else:
            dy, dx = allocator.allocate(w, h)
            allocs[i][j] = (dy, dx)

            packed[dy : dy + h, dx : dx + w] = src[sy : sy + h, sx : sx + w]

    print(f"Packing '{output}' finished, took {time() - t0}s")

    packed = packed[0 : allocator.num_used_rows]

    packed_file = f"{output}_packed.png"
    imageio.imsave(packed_file, packed)

    # Generate JSON to represent the data
    delays = (array(times[1:] + [times[-1] + end_pause]) - array(times)).tolist()

    timeline = []
    for i in range(len(images)):
        src_rects = img_areas[i]
        dst_rects = allocs[i]

        blitlist = []

        for j in range(len(src_rects)):
            a, b = src_rects[j]
            sx, sy = b.start, a.start
            w, h = b.stop - b.start, a.stop - a.start
            dy, dx = dst_rects[j]

            blitlist.append([dx, dy, w, h, sx, sy])

        timeline.append({"delay": delays[i], "blit": blitlist})

    meta_file = f"{output}_anim.{output_format}"
    with open(meta_file, "wb") as f:
        # write a variable declaration
        if output_format == "js":
            f.write(f"{output}_timeline=".encode("utf-8"))

        f.write(json.dumps(to_native(timeline), separators=(",", ":")).encode("utf-8"))

    return packed_file, meta_file


def optimize_command(command, input_file, output_file):
    """
    Runs a command with the input and output file then
    overwrites the output file to the input file.
    """
    try:
        print(f"Compressing file with {command[0]}")
        check_call(
            [x.format(input_file, output_file) for x in command],
            stdout=DEVNULL,
            stderr=STDOUT,
        )
        shutil.move(output_file, input_file)
    except FileNotFoundError:
        print(f"{command[0]} not found, cannot reduce filesize losslessly")


def optimize_lossless(input_file, command=[]):
    """
    Optimizes the input file with pngcrush.
    """
    output_file = f"{input_file}_lossless.png"
    optimize_command(command, input_file, output_file)


def optimize_lossy(input_file):
    """
    Optimizes the input file with pngquant.
    pngquant may significantly reduce filesize for screencasts
    that don't include photos or other sources of many different colors.
    """
    output_file = f"{input_file}_lossy.png"
    optimize_command(["pngquant", "-o", "{1}", "{0}"], input_file, output_file)


if __name__ == "__main__":
    args = parse_args()

    input_files = []
    output = args.output
    if output is None:
        # use the directory name or filename (without extension) for output
        output = (
            args.animation_path[0]
            if os.path.isdir(args.animation_path[0])
            else os.path.splitext(args.animation_path[0])[0]
        )

    for anim_path in args.animation_path:
        if os.path.isdir(anim_path):
            # append file inside the directory into the list
            for f in os.listdir(anim_path):
                input_files.append(os.path.join(anim_path, f))
        else:
            # append the file as-is
            input_files.append(anim_path)
    input_files.sort()

    packed_filename, _ = generate_animation(
        input_files, output, args.fps, args.end_pause, args.format
    )

    if args.compress:
        optimize_lossless(packed_filename, shlex.split(args.compress_command))
    if args.compress_pngquant:
        optimize_lossy(packed_filename)
