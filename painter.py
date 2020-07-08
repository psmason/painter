import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import random
import math
import argparse

GRIDS = 12
HEIGHT = 4000
WIDTH = 3000
COLORS_IN_PALETTE = 8
BRUSH_INTENSITY_THRESHOLD = 0.05

REVERTED_COUNT = 0

def extractBrushStrokesFromGrayscale(img):
    denoisedAndInverted = cv2.medianBlur(255-img, 5)

    # Identifying black pigments distinct from the nearby
    # regions background color.
    background_colors = np.zeros(img.shape, np.uint8)
    yDim, xDim = img.shape
    for i in range(GRIDS):
        yRange = int(i*(yDim/GRIDS)),int((i+1)*(yDim/GRIDS))
        for j in range(GRIDS):
            xRange = int(j*(xDim/GRIDS)),int((j+1)*(xDim/GRIDS))
            m = np.median(denoisedAndInverted[yRange[0]:yRange[1], xRange[0]:xRange[1]])
            background_colors[yRange[0]:yRange[1], xRange[0]:xRange[1]] = m
    # Masking pixels which are similar to the nearby region's background color.
    mask = np.greater(denoisedAndInverted, background_colors+30)

    # Dilating the mask to avoid cutting off segments of the brush stroke.
    dilated_mask = cv2.dilate(mask.astype(np.uint8), np.ones((5,5),np.uint8), iterations = 15)

    # Identifying distinct brush strokes from other strokes.
    _, contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    brush_strokes = []
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])

        if (x == 0 or x == img.shape[1]) or (y == 0 or y == img.shape[0]):
            # Bounding rectangle is on the image edge. Probably noise.
            continue

        raw_brush_stroke = denoisedAndInverted[y:y+h,x:x+w]

        # Using edges to describe background color.
        edge_median = np.median(
            np.concatenate(
                (raw_brush_stroke[:,0],
                 raw_brush_stroke[0,:],
                 raw_brush_stroke[:,-1],
                 raw_brush_stroke[-1,:]), axis=None)
        )

        # A brush stroke is described by a matrix of color intensities.
        # An intensity of 1.0 is the original color. An intensity closer
        # to 0.0 is original color transformed to value closer to white.
        normalized = abs(edge_median - raw_brush_stroke)
        normalized = normalized / np.max(normalized)

        # Zero'ing values below a threshold.
        normalized[normalized < BRUSH_INTENSITY_THRESHOLD] = 0

        brush_strokes.append(normalized)

    return brush_strokes

def extractColorPalette(target):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _,_,palette = cv2.kmeans(np.float32(target.reshape((-1,1))),
                                  COLORS_IN_PALETTE,
                                  None,
                                  criteria,
                                  10,
                                  cv2.KMEANS_RANDOM_CENTERS)
    return palette

def randomBrushStroke(target, canvas, brush_strokes, palette, debug_mode):
    def computeImageDistance(img1, img2, weights):
        return np.sum(np.multiply(np.square(img1-img2),weights))

    global REVERTED_COUNT
    brush = random.choice(brush_strokes)

    rotation, scale = random.randint(0, 365), random.uniform(0.5, 1.0)
    rotated = ndimage.rotate(brush, rotation, reshape=True)
    scaled = cv2.resize(rotated, (0,0), fx=scale, fy=scale)

    random_color = random.choice(palette)
    #random_color = (random.randint(0,255), random.randint(0, 255), random.randint(0,255))
    #b = random_color[0] + (255.0-random_color[0]) * (1.0-scaled)
    #g = random_color[1] + (255.0-random_color[1]) * (1.0-scaled)
    #r = random_color[2] + (255.0-random_color[2]) * (1.0-scaled)
    #merged = cv2.merge((b.astype(np.uint8),g.astype(np.uint8),r.astype(np.uint8)))

    # Random brush location that can contain the brush stroke.
    location = (random.randint(0, canvas.shape[0] - scaled.shape[0] - 1),
                random.randint(0, canvas.shape[1] - scaled.shape[1] - 1))

    rowsToUpdate = (location[0], location[0]+scaled.shape[0])
    colsToUpdate = (location[1], location[1]+scaled.shape[1])

    canvas_roi = canvas[rowsToUpdate[0]:rowsToUpdate[1],colsToUpdate[0]:colsToUpdate[1]]
    target_roi = target[rowsToUpdate[0]:rowsToUpdate[1],colsToUpdate[0]:colsToUpdate[1]]
    canvas_roi_original = canvas_roi.copy()
    distance_original = computeImageDistance(target_roi, canvas_roi_original, scaled)

    # linear blend onto the canvas. blend of the new brush stroke depends on the brush pixel intensity.
    #canvas_roi[:,:,0] = np.multiply(1.0 - scaled_roi, canvas_roi[:,:,0])
    #canvas_roi[:,:,1] = np.multiply(1.0 - scaled_roi, canvas_roi[:,:,1])
    #canvas_roi[:,:,2] = np.multiply(1.0 - scaled_roi, canvas_roi[:,:,2])
    #canvas_roi[:,:,0] += np.multiply(scaled_roi, brush_roi[:,:,0])
    #canvas_roi[:,:,1] += np.multiply(scaled_roi, brush_roi[:,:,1])
    #canvas_roi[:,:,2] += np.multiply(scaled_roi, brush_roi[:,:,2])
    canvas_roi = random_color + np.multiply(canvas_roi-random_color, 1.0-scaled)
    distance_new = computeImageDistance(target_roi, canvas_roi, scaled)
    
    if debug_mode:
        random_brush_stroke = random_color + (255.0-random_color)*(1.0-scaled)
        fig, axs = plt.subplots(3,3, figsize=(10,10))
        axs[0][0].set_title('canvas')
        axs[0][0].imshow(canvas, cmap='gray', vmin=0, vmax=255)
        axs[0][1].set_title('canvas roi: d = %3.2f' % (math.log(distance_new)))
        axs[0][1].imshow(canvas_roi, cmap='gray', vmin=0, vmax=255)
        axs[0][2].set_title('canvas original roi: d = %3.2f' % (math.log(distance_original)))
        axs[0][2].imshow(canvas_roi_original, cmap='gray', vmin=0, vmax=255)
        axs[1][0].set_title('target image')
        axs[1][0].imshow(target, cmap='gray', vmin=0, vmax=255)
        axs[1][1].set_title('target roi')
        axs[1][1].imshow(target_roi, cmap='gray', vmin=0, vmax=255)
        axs[1][2].set_title('raw brush stroke')
        axs[1][2].imshow(brush, cmap='gray')
        axs[2][0].set_title('brush weights')
        axs[2][0].imshow(scaled, cmap='gray')
        axs[2][1].set_title('scaled brush: color = %d' % (random_color))
        axs[2][1].imshow(np.multiply(scaled, random_brush_stroke), cmap='gray', vmin=0, vmax=255)
        axs[2][2].set_title('brush stroke: color = %d' % (random_color))
        axs[2][2].imshow(random_brush_stroke, cmap='gray', vmin=0, vmax=255)
        plt.show()

    if distance_new < distance_original:
        canvas[rowsToUpdate[0]:rowsToUpdate[1],colsToUpdate[0]:colsToUpdate[1]] = canvas_roi
    else:
        # Avoiding the update
        REVERTED_COUNT += 1

parser = argparse.ArgumentParser(description='Reconstruct an image using extracted brush strokes.')
parser.add_argument('--iterations', type=int, help='Brush stroke iterations to run')
parser.add_argument('--brushes_image', type=str, help='Source image for brush strokes')
parser.add_argument('--target_image', type=str, help='Target image to reconstruct')
parser.add_argument('--output_image_name', type=str, help='Output file name')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

brush_strokes = extractBrushStrokesFromGrayscale(cv2.imread(args.brushes_image, cv2.IMREAD_GRAYSCALE))
target = cv2.imread(args.target_image,  cv2.IMREAD_GRAYSCALE)
color_palette = extractColorPalette(target)
target_float = target.astype(np.float)
canvas = 255 * np.ones(target.shape, np.float)

for i in range(args.iterations):
    randomBrushStroke(target_float, canvas, brush_strokes, color_palette, args.debug)
    if i > 0 and i % 1000 == 0:
        print("iteration:", i, "reverted fraction:", 1.0*REVERTED_COUNT/i)

cv2.imwrite(args.output_image_name, canvas.astype(np.uint8))
plt.imshow(canvas, cmap='gray', vmin=0, vmax=255)
plt.show()
