#!/usr/bin/env python3

import argparse
import cv2
import math
import numpy as np
import random
import sys

from matplotlib import pyplot as plt
from scipy import ndimage

GRIDS = 12
COLORS_IN_PALETTE = 256
BRUSH_INTENSITY_THRESHOLD = 0.05
MIN_BRUSH_SCALE = 0.15
MAX_BRUSH_SCALE = 1.0


def extract_brush_strokes_from_grayscale(img, debug_mode):
    denoisedAndInverted = cv2.medianBlur(255 - img, 5)

    # Identifying black pigments distinct from the local background color.
    background_colors = np.zeros(img.shape, np.uint8)
    yDim, xDim = img.shape
    for i in range(GRIDS):
        yRange = int(i * (yDim / GRIDS)), int((i + 1) * (yDim / GRIDS))
        for j in range(GRIDS):
            xRange = int(j * (xDim / GRIDS)), int((j + 1) * (xDim / GRIDS))
            m = np.median(
                denoisedAndInverted[yRange[0]:yRange[1], xRange[0]:xRange[1]])
            background_colors[yRange[0]:yRange[1], xRange[0]:xRange[1]] = m
    # Masking pixels which are similar to the local background color.
    mask = np.uint8(np.greater(denoisedAndInverted, background_colors + 30))

    # Dilating the mask to avoid cutting off segments of the brush stroke.
    dilated_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=15)

    # Identifying distinct brush strokes from other strokes.
    _, contours, _ = cv2.findContours(
        dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug_mode:
        tmp = img.copy()
        cv2.drawContours(tmp, contours, -1, 255, 10)
        plot_debug_image(plt, tmp)
        plt.title("Candidate brush regions")
        plt.show()

    bounding_rects = []
    brush_strokes = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])

        if (x == 0 or x == img.shape[1]) or (y == 0 or y == img.shape[0]):
            # Bounding rectangle is on the image edge. Probably noise.
            continue

        raw_brush_stroke = denoisedAndInverted[y:y + h, x:x + w]

        # Using edges to describe background color.
        edge_median = np.median(
            np.concatenate(
                (raw_brush_stroke[:, 0],
                 raw_brush_stroke[0, :],
                 raw_brush_stroke[:, -1],
                 raw_brush_stroke[-1, :]), axis=None)
        )

        # Intensity as described by the color distance from the brush stroke
        # border.
        intensities = abs(edge_median - raw_brush_stroke)
        hist = np.histogram(intensities, np.arange(256))
        # Does the intensity image have enough pixels significantly different from the
        # background color. If not, this might be a false region for a brush
        # stroke.
        # Or if the diff is too small, might be a speck of dust or paint.
        if np.sum(hist[0][50:]) <= 100:
            continue

        bounding_rects.append((x, y, w, h))
        normalized = intensities / np.max(intensities)

        # Zero'ing values below a threshold.
        normalized[normalized < BRUSH_INTENSITY_THRESHOLD] = 0

        # Rotations are expensive, so let's do them once and keep the result
        # in memory.
        for angle in range(0, 360, 10):
            rotated = ndimage.rotate(normalized, angle, reshape=True)
            # Rotations introduce negative intensities, probably because of rotation
            # matrices involved: https://en.wikipedia.org/wiki/Rotation_matrix.
            # If not removed, negative intensities cause problem when multiplied
            # against a random color.
            rotated[rotated < 0.0] = 0
            brush_strokes.append(rotated)

    if debug_mode:
        tmp = img.copy()
        for rect in bounding_rects:
            x, y, w, h = rect
            cv2.rectangle(tmp, (x, y), (x + w, y + h), 255, 10)
        plt.title("Used brush strokes")
        plot_debug_image(plt, tmp)
        plt.show()

    return brush_strokes


def get_color_depth(img):
    dims = img.shape
    if len(dims) == 2:
        # Simple black and white
        return 1
    if len(dims) == 3:
        return dims[-1]
    print("Unexpected color space", file=sys.stderr)
    sys.exit(1)


def plot_debug_image(axis, img, **kwargs):
    if get_color_depth(img) == 3:
        axis.imshow(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB))
        return

    # Black and white.
    if len(img.shape) == 3:
        # Truncate trailing dimension.
        img = img[:, :, 0]
    if kwargs.get('scaled', False):
        axis.imshow(img, cmap='gray')
    else:
        axis.imshow(np.uint8(img), cmap='gray', vmin=0, vmax=255)


def extract_color_palette(target, debug_mode):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, palette = cv2.kmeans(np.float32(target.reshape((-1, get_color_depth(target)))),
                                    COLORS_IN_PALETTE,
                                    None,
                                    criteria,
                                    10,
                                    cv2.KMEANS_PP_CENTERS)

    if debug_mode:
        res = palette[labels.flatten()]
        reduced_palette = res.reshape((target.shape))
        fig, axs = plt.subplots(2, figsize=(10, 10))
        axs[0].set_title("Target image")
        plot_debug_image(axs[0], target)
        axs[1].set_title("Reduced palette")
        plot_debug_image(axs[1], reduced_palette)
        plt.show()

    return palette


def random_brush_stroke(
        target,
        canvas,
        brush_strokes,
        palette,
        counters,
        debug_mode):
    def compute_image_distance(img1, img2, weights):
        return np.sum(np.multiply(np.square(img1 - img2), weights))

    brush = random.choice(brush_strokes)

    # Distribution favoring smaller brush strokes, which are needed for detailed
    # image sections. Resulting range is 0.1 to 1.0 of the original brush
    # stroke.
    scale = 1.0 / random.uniform(1.0 / MAX_BRUSH_SCALE, 1.0 / MIN_BRUSH_SCALE)
    scaled = cv2.resize(brush, None, fx=scale, fy=scale)

    # Empty trailing dimension needed for numpy broadcasting.
    scaled = scaled[..., np.newaxis]

    if canvas.shape[0] <= scaled.shape[0] or canvas.shape[1] <= scaled.shape[1]:
        # Scaled brush stroke is too large.
        return

    # Random brush location that can contain the brush stroke.
    location = (random.randint(0, canvas.shape[0] - scaled.shape[0] - 1),
                random.randint(0, canvas.shape[1] - scaled.shape[1] - 1))

    rowsToUpdate = (location[0], location[0] + scaled.shape[0])
    colsToUpdate = (location[1], location[1] + scaled.shape[1])

    canvas_roi = canvas[rowsToUpdate[0]:rowsToUpdate[1],
                        colsToUpdate[0]:colsToUpdate[1]]
    target_roi = target[rowsToUpdate[0]:rowsToUpdate[1],
                        colsToUpdate[0]:colsToUpdate[1]]
    canvas_roi_original = canvas_roi.copy()
    distance_original = compute_image_distance(
        target_roi, canvas_roi_original, scaled)

    # Linear blend onto the canvas, with blend weights coming from brush stroke intensities.
    # Intensity closer to 1.0 favors the brush stroke color. Intensity closer to 0.0 favors
    # the background canvas color.
    random_color = random.choice(palette)
    canvas_roi = random_color + \
        np.multiply(canvas_roi - random_color, 1.0 - scaled)
    distance_new = compute_image_distance(target_roi, canvas_roi, scaled)

    if debug_mode:
        random_brush_on_white = random_color + \
            (255.0 - random_color) * (1.0 - scaled)
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        axs[0][0].set_title('canvas')
        plot_debug_image(axs[0][0], canvas)
        axs[0][1].set_title('canvas roi: d = %3.2f' % (math.log(distance_new)))
        plot_debug_image(axs[0][1], canvas_roi)
        axs[0][2].set_title(
            'canvas original roi: d = %3.2f' %
            (math.log(distance_original)))
        plot_debug_image(axs[0][2], canvas_roi_original)
        axs[1][0].set_title('target image')
        plot_debug_image(axs[1][0], target)
        axs[1][1].set_title('target roi')
        plot_debug_image(axs[1][1], target_roi)
        axs[1][2].set_title('raw brush stroke')
        plot_debug_image(axs[1][2], brush, scaled=True)
        axs[2][0].set_title('brush weights')
        plot_debug_image(axs[2][0], scaled, scaled=True)
        axs[2][1].set_title('scaled brush')
        plot_debug_image(axs[2][1], np.multiply(scaled, random_brush_on_white))
        # Reversing colors since matplotlib is RGB.
        axs[2][2].set_title('brush stroke: color = %s' %
                            (str(np.uint8(random_color[::-1]))))
        plot_debug_image(axs[2][2], random_brush_on_white)
        plt.show()

    if distance_new < distance_original:
        # Applying the update
        canvas[rowsToUpdate[0]:rowsToUpdate[1],
               colsToUpdate[0]:colsToUpdate[1]] = canvas_roi
        counters[1] += 1
    else:
        # Rejecting the brush stroke.
        counters[0] += 1


parser = argparse.ArgumentParser(
    description='Reconstruct an image using extracted brush strokes.')
parser.add_argument(
    '--iterations',
    type=int,
    help='Brush stroke iterations to run')
parser.add_argument(
    '--brushes_image',
    type=str,
    help='Source image for brush strokes')
parser.add_argument(
    '--target_image',
    type=str,
    help='Target image to reconstruct')
parser.add_argument('--output_image_name', type=str, help='Output file name')
parser.add_argument(
    '--output_interval',
    type=int,
    default=0,
    help='Interval between iterations to print canvas to stdout (jpg)')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument(
    '--bw',
    action='store_true',
    help='Enable black and white mode')
args = parser.parse_args()

print("Extracting brush strokes...", file=sys.stderr)
brush_strokes = extract_brush_strokes_from_grayscale(
    cv2.imread(args.brushes_image, cv2.IMREAD_GRAYSCALE),
    args.debug)
target = cv2.imread(args.target_image, cv2.IMREAD_COLOR)
if args.bw:
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # Trailing axis needed for numpy broadcasting.
    target = target[..., np.newaxis]

print("Extracting color palette...", file=sys.stderr)
color_palette = extract_color_palette(target, args.debug)

print("Painting...", file=sys.stderr)
target_float = target.astype(np.float)
canvas = 255.0 * np.ones(target.shape, np.float)
counters = [
    0,  # Recently rejected count
    0  # Accepted count
]
for i in range(args.iterations):
    random_brush_stroke(
        target_float,
        canvas,
        brush_strokes,
        color_palette,
        counters,
        args.debug)
    if i > 0:
        if i % 1000 == 0:
            print("iteration:", i, "recently reverted fraction:", 1.0 * counters[0] / 1000, file=sys.stderr)
            counters[0] = 0
        if args.output_interval > 0 and counters[1] % args.output_interval == 0:
            sys.stdout.buffer.write(cv2.imencode(".jpg", canvas)[1].tostring())

if args.debug:
    plot_debug_image(plt, canvas)
    plt.show()

canvas = np.uint8(canvas)
if args.bw:
    # Removing trailing axis needed for numpy broadcasting.
    canvas = canvas[:, :, 0]

cv2.imwrite(args.output_image_name, canvas)
