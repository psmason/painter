import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import random

GRIDS = 12
HEIGHT = 4000
WIDTH = 3000
ALPHA = 0.7

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

        #plt.imshow(denoisedAndInverted[y:y+h,x:x+w], cmap='gray')
        #plt.show()
    
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
        brush_strokes.append(abs(edge_median - raw_brush_stroke)/255.0)

    return brush_strokes

def randomBrushStroke(target, canvas, brush_strokes):
    global REVERTED_COUNT
    brush = random.choice(brush_strokes)

    rotation, scale = random.randint(0, 365), random.uniform(0.25, 1.0)
    rotated = ndimage.rotate(brush, rotation, reshape=True)
    scaled = cv2.resize(rotated, (0,0), fx=scale, fy=scale)

    random_color = (random.randint(0,255), random.randint(0, 255), random.randint(0,255))
    b = random_color[0] + (255-random_color[0]) * (1.0-scaled)
    g = random_color[1] + (255-random_color[1]) * (1.0-scaled)
    r = random_color[2] + (255-random_color[2]) * (1.0-scaled)
    merged = cv2.merge((b.astype(np.uint8),g.astype(np.uint8),r.astype(np.uint8)))

    # linear blend onto the canvas. blend of the new brush stroke depends on the brush pixel intensity.
    location = random.randint(0, canvas.shape[0]), random.randint(0, canvas.shape[1])
    rowsToUpdate = (location[0], min(location[0]+merged.shape[0], canvas.shape[0]-1))
    colsToUpdate = (location[1], min(location[1]+merged.shape[1], canvas.shape[1]-1))

    brushExtractedHeight = rowsToUpdate[1]-rowsToUpdate[0]
    brushExtractedWidth = colsToUpdate[1]-colsToUpdate[0]
    if brushExtractedHeight <= 0 or brushExtractedWidth <= 0:
        return

    canvas_roi = canvas[rowsToUpdate[0]:rowsToUpdate[1],colsToUpdate[0]:colsToUpdate[1]]
    brush_roi = merged[:brushExtractedHeight,:brushExtractedWidth]
    scaled_roi = scaled[:brushExtractedHeight,:brushExtractedWidth]

    target_roi = target[rowsToUpdate[0]:rowsToUpdate[1],colsToUpdate[0]:colsToUpdate[1]]
    canvas_roi_original = canvas_roi.copy()
    distance_original = np.linalg.norm(target_roi - canvas_roi_original)

    canvas_roi[:,:,0] = np.multiply(1.0 - scaled_roi, canvas_roi[:,:,0]).astype(np.uint8)
    canvas_roi[:,:,1] = np.multiply(1.0 - scaled_roi, canvas_roi[:,:,1]).astype(np.uint8)
    canvas_roi[:,:,2] = np.multiply(1.0 - scaled_roi, canvas_roi[:,:,2]).astype(np.uint8)
    canvas_roi[:,:,0] += np.multiply(scaled_roi, brush_roi[:,:,0]).astype(np.uint8)
    canvas_roi[:,:,1] += np.multiply(scaled_roi, brush_roi[:,:,1]).astype(np.uint8)
    canvas_roi[:,:,2] += np.multiply(scaled_roi, brush_roi[:,:,2]).astype(np.uint8)
    distance_new = np.linalg.norm(target_roi - canvas_roi)

    if distance_original < distance_new:
        # revert the change
        REVERTED_COUNT += 1
        canvas[rowsToUpdate[0]:rowsToUpdate[1],colsToUpdate[0]:colsToUpdate[1]] = canvas_roi_original

brush_strokes = extractBrushStrokesFromGrayscale(cv2.imread("brush_strokes.jpg", cv2.IMREAD_GRAYSCALE))
target = cv2.imread("portrait.jpg",  cv2.IMREAD_COLOR)
canvas = 255 * np.ones(target.shape, np.uint8)

for i in range(1000*1000):
    randomBrushStroke(target, canvas, brush_strokes)
    if i > 0 and i % 1000 == 0:
        print("iteration:", i, "reverted fraction:", 1.0*REVERTED_COUNT/i)

    if i > 1000 and 1.0*REVERTED_COUNT/i > 0.98:
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.show()


