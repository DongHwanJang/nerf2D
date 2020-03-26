import glob
import os

raw_xy  = sorted(glob.glob("./dataset/cool_cows/*raw_xy*.jpg"))
sin_cos = sorted(glob.glob("./dataset/cool_cows/*sin_cos*.jpg"))

count = 0
for x, y in zip(raw_xy, sin_cos):
    print(count, x, y)
    outfileName = 'evolution_{:04d}.jpg'.format(int(count))
    os.system("montage ./dataset/cool_cows.jpg {} {} -geometry +0+0 {}".format(y, x, outfileName))

    count += 1 