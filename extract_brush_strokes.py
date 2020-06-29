import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

original_image = cv2.imread("brush_strokes.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.medianBlur(255-original_image, 5)

plt.imshow(img, cmap='gray')
plt.show()

background_colors = np.zeros(img.shape, np.uint8)
yDim, xDim = img.shape
GRIDS = 12
for i in range(GRIDS):
    yRange = int(i*(yDim/GRIDS)),int((i+1)*(yDim/GRIDS))
    for j in range(GRIDS):
        xRange = int(j*(xDim/GRIDS)),int((j+1)*(xDim/GRIDS))
        m = np.median(img[yRange[0]:yRange[1], xRange[0]:xRange[1]])
        background_colors[yRange[0]:yRange[1], xRange[0]:xRange[1]] = m
mask = np.greater(img, background_colors+30)

#opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
#plt.show()
dilated_mask = cv2.dilate(mask.astype(np.uint8), np.ones((5,5),np.uint8), iterations = 15)
#plt.imshow(dilated_mask, cmap='gray')
#plt.show()

img_masked = cv2.bitwise_and(img, dilated_mask*255)
#plt.imshow(img_masked, cmap='gray')
#plt.show()

#_, thresh = cv2.threshold(dilation, 50, 255, cv2.THRESH_BINARY)
#plt.imshow(thresh, cmap='gray')
#plt.show()

_, contours, _ = cv2.findContours(img_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))


brush_strokes = []
for i in range(len(contours)):
    x,y,w,h = cv2.boundingRect(contours[i])
    #print(x,y,w,h)

    brush_stroke = original_image[y:y+h,x:x+w]
    brush_strokes.append(brush_stroke)

    edge_median = np.median(np.concatenate((brush_stroke[:,0], brush_stroke[0,:], brush_stroke[:,-1], brush_stroke[-1,:]), axis=None))
    print(edge_median)

    plt.imshow(brush_stroke, cmap='gray')
    plt.show()

    random_color = (random.randint(0,255), random.randint(0, 255), random.randint(0,255))
    # how far to scale the color to white.
    diffs = (255-abs(edge_median - brush_stroke))/255.0
    b = random_color[0] + (255-random_color[0]) * diffs
    g = random_color[1] + (255-random_color[1]) * diffs
    r = random_color[2] + (255-random_color[2]) * diffs
    merged = cv2.merge((b.astype(np.uint8),g.astype(np.uint8),r.astype(np.uint8)))
    plt.imshow(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
    plt.show()
