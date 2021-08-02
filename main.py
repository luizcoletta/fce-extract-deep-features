import arff  # Library https://pypi.org/project/liac-arff/. For read/write arff files
import cv2
import numpy as np
import os

# https://stackoverflow.com/questions/3430372/how-do-i-get-the-full-path-of-the-current-files-directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

data_file_name = "ceratocystis_test_90_10_rgb_11_fs9_res1-Indexes.arff"
data_path = curr_dir + "/data/" + data_file_name

data = arff.load(open(data_path))

# Convert dicc to list. Only the data
lst_data = list(data.values())[3]

# Loop to search one register of listNoCoord in listWithCoord
'''for l in lst_data:
    first_or_default = next((x for x in listWithCoord if x[3] == l[0] and x[4] == l[1] and x[5] == l[2]), None)
    if first_or_default is not None:
        #list to be appended to the listMatched. Contains the number of image, coord x, coord y, real class, predicted class
        abc = [first_or_default[0], first_or_default[1], first_or_default[2], int(first_or_default[-1]), int(l[-1])]
        listMatched.append(abc)

#Write of a .txt file with the listMatched
with open('imageCoordTo'+ResultFile+'.txt', 'w') as f:
    for item in listMatched:
        f.write("%4.0f\t%4.0f\t%4.0f\t%s\t%s\n" % (item[0], item[1], item[2], item[3], item[4]))'''



# https://learnopencv.com/cropping-an-image-using-opencv/
img = cv2.imread('images/test.png')
print(img.shape)  # Print image shape
cv2.imshow("original", img)

# Cropping an image
cropped_image = img[80:280, 150:330]  # img[start_row:end_row, start_col:end_col]

# Display cropped image
cv2.imshow("cropped", cropped_image)

# Save the cropped image
cv2.imwrite("Cropped Image.png", cropped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()