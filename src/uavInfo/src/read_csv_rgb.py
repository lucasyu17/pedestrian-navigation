import cv2
import numpy as np
import sys
import csv
import os

img_wid = 256
img_height = 192
filename_rgb_csv = "/home/ubuntu/chg_workspace/data/new_map_with_depth_img/depth_rgb_semantics/gazebo_rate_092/yhz/short/20/rgb_data_2019_03_09_18:42:31.csv"
rgb_imgs = open(filename_rgb_csv, "r")
img_num = len(rgb_imgs.readlines())


def read_rgb_csv(filename, data):
    maxInt = sys.maxsize
    decrement = True
    try:
        csv.field_size_limit(maxInt)
        with open(filename, mode='r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            i_row = 0
            for row in csv_reader:
                for i in range(img_height):
                    for j in range(img_wid):
                        for k in range(3):
                            data[i_row, i, j, k] = row[i * img_wid * 3 + j * 3 + k]
                i_row = i_row + 1
    except OverflowError:
        maxInt = int(maxInt / 10)
        decrement = True


if __name__ == "__main__":
    # numpy array [img_number * img_width * img_height * rgb]
    np_data_rgb = np.zeros([img_num, img_height, img_wid, 3])
    # read data
    print "reading rgb data from csv file: %s..." % filename_rgb_csv
    read_rgb_csv(filename_rgb_csv, np_data_rgb)
    print "wrting rgb img files to ./testdata/img/"
    save_path = "testdata/img"
    for i in range(img_num):
        img_this = np_data_rgb[i]
        print("img size: " + str(img_this.shape))
        cv2.imshow("testshow", img_this)
        file_name = 'color_img_%d.jpg' % i
        save_file = os.path.join(save_path, file_name)
        print save_file
        cv2.imwrite(save_file, img_this)
