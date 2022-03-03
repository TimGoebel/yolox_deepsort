

def vid_to_frames(path):
    frames = []
    cap = cv2.VideoCapture(path)
    ret = True
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            frames.append(img)
    return frames

# Overlaying Image and Emoji
def add_image(img, src2, x, y):
    # x=  x+90
    # y = y-10
    w = 80
    h = 80

    initial = img[y:y+h,x:x+w]
    src1 = initial
    src2 = cv2.resize(src2, src1.shape[1::-1])
    u_green = np.array([1, 1, 1])
    l_green = np.array([0, 0, 0])
    mask = cv2.inRange(src2, l_green, u_green)
    res = cv2.bitwise_and(src2, src2, mask = mask)
    f = src2 - res
    f = np.where(f == 0, src1, f)
    img[y:y+h,x:x+w] = f
    return img


emojidict = dict(
    car = vid_to_frames('assets/car.gif'),
    truck = vid_to_frames('assets/truck.gif'),   

    )