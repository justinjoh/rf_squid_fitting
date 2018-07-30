import cv2
import numpy as np

randmat = np.random.rand(100, 100)
# randcv = cv2.resize(randmat, (100, 100))
cv2.imshow('aaa', randmat)
cv2.waitKey(0)
cv2.destroyAllWindows()
