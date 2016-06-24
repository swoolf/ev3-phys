#!/usr/bin/env python

# Hard coding these values is not a good idea because the values could
# change. But, since this is an example, we want to keep it short.
SCREEN_WIDTH = 178 # pixels
SCREEN_HEIGHT = 128 # pixels
LINE_LENGTH = 24 # bytes
SIZE = 3072 # bytes

import os
import array


def drawIf(buf):

    # draw a vertical lines in column 100 (0 based index)
    for col in (30,31,32, 45,46,47):
        for row in range(30, SCREEN_HEIGHT-30):
            buf[row * LINE_LENGTH + int(col / 8)] = 1 << (col % 8)

    # draw a horizontal line in row 64 (0 based index)
    for row2 in (30,60):
        for col2 in range(55, 65):
            buf[row2 * LINE_LENGTH + int(col2/8)] = 0xff
    return buf

def drawThen(buf):
    
    # draw a vertical lines in column 100 (0 based index)
    for col in (30,31,32):
        for row in range(30, SCREEN_HEIGHT-30):
            buf[row * LINE_LENGTH + int(col / 8)] = 1 << (col % 8)

    # draw a horizontal line in row 64 (0 based index)
    for row2 in (30,31,32):
        for col2 in range(20, 45):
            buf[row2 * LINE_LENGTH + int(col2/8)] = 0xff

    return buf


def drawz():
    buf = [0] * SIZE
    
    # draw a vertical line in column 100 (0 based index)
    for row in range(0, SCREEN_HEIGHT):
        buf[row * LINE_LENGTH + int(100 / 8)] = 1 << (100 % 8)
    
    # draw a horizontal line in row 64 (0 based index)
    for col in range(0, LINE_LENGTH):
        buf[64 * LINE_LENGTH + col] = 0xff
    
    
    import math
    # draw a circle, center at (40,40), radius is 20
    for x in range(0, 20):
        y = math.sqrt(20 * 20 - x * x)
        buf[(40 + int(y)) * LINE_LENGTH + int((40 + x) / 8)] = 1 << ((40 + x) % 8)
        buf[(40 - int(y)) * LINE_LENGTH + int((40 + x) / 8)] = 1 << ((40 + x) % 8)
        buf[(40 + int(y)) * LINE_LENGTH + int((40 - x) / 8)] = 1 << ((40 - x) % 8)
        buf[(40 - int(y)) * LINE_LENGTH + int((40 - x) / 8)] = 1 << ((40 - x) % 8)

def drawNow(state):
    buf = [0] * SIZE
    if state == 'then':
        drawThen(buf)
    elif state =='if':
        drawIf(buf)
    
    f = os.open('/dev/fb0', os.O_RDWR)
    s = array.array('B', buf).tostring()
    os.write(f, s)
    os.close(f)

