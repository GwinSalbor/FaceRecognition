import os
import dlib
from PIL import Image
from skimage import io
from scipy.spatial import distance

sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
img = io.imread('You.jpg') #add your picture

win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(img)
dets = detector(img, 1)

for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)
face_descriptor1 = facerec.compute_face_descriptor(img, shape)
print(face_descriptor1)

directory = '' # add directory with your .jpg files
files = os.listdir(directory)
images = filter(lambda x: x.endswith('.jpg'), files)

for i in images:
    img = io.imread(directory + '\\' + i)
    win2 = dlib.image_window()
    win2.clear_overlay()
    win2.set_image(img)
    dets_webcam = detector(img, 1)
    for k, d in enumerate(dets_webcam):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        shape = sp(img, d)
        win2.clear_overlay()
        win2.add_overlay(d)
        win2.add_overlay(shape)
    face_descriptor2 = facerec.compute_face_descriptor(img, shape)
    a = distance.euclidean(face_descriptor1, face_descriptor2)
    wait = input("PRESS ENTER TO CONTINUE.")

    print(a)
    if( a < 0.55):
        image = Image.open("C:\\Users\\Yaroslav\\Desktop\\Foto\\"+ i)
        image.save("C:\\Users\\Yaroslav\\Desktop\\Foto\\out\\" + i, "JPEG")
