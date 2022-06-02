import cv2
import dlib
import numpy as np
from keras.models import load_model
from imutils import face_utils
import pyautogui
import keyboard
import turtle
from turtle import Screen
import time
import datetime

# köşelerde kapanmasın
pyautogui.FAILSAFE = False

# gerekli dosyaların adreslerini tanımlama
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("kaynak\shape_predictor.dat")
face_cascade = cv2.CascadeClassifier('kaynak\haarcascade_frontalface.xml')

# sol sağ göz varsayılan noktaları
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]


def eye_on_mask(shape, mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def contouring(thresh, mid, img, right_value=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right_value:
            cx += mid
        # göz kısmının etrafında noktalar çizilir.
        cv2.circle(img, (cx, cy), 10, (255, 255, 255), 1)
        return cx, cy
    except:
        pass


# yüzü algıla
def detect(img, minimumFeatureSize=(20, 20)):
    if face_cascade.empty():
        raise (Exception('haar Cascade xml dosyası yüklenirken hata.'))
    rects = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)

    # yüz bulunmazsa diziyi 0 değeriyle yolla
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


# göz kısımlarını kırp
def crop_eyes(frame):
    global face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # gri resimden yüzü algıla
    te = detect(gray, minimumFeatureSize=(80, 80))

    if len(te) == 0:  # eğer yüz algılanmazsa 0 değeri döndür
        return None
    elif len(te) > 1:  # 1den fazla yüz algılanırsa tekrar işlem yap
        face = te[0]
    elif len(te) == 1:
        [face] = te

    # görüntünün 4 tarafından yüz indexleriyle kareden sadece yüz olan kısmı tut
    face_rect = dlib.rectangle(left=int(face[0]), top=int(face[1]),
                               right=int(face[2]), bottom=int(face[3]))

    # yüz tipini belirlemek için dlib shape predictor dosyası kullanılır
    shape = predictor(gray, face_rect)
    shape = face_utils.shape_to_np(shape)

    # sağ ve sol göz'ün baş ve son indexlerini al
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # sağ ve sol gözün koordinatlarını çıkar
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    # sol gözün alt ve üst değerini alıp yüksekliğini hesapla
    l_uppery = min(leftEye[1:3, 1])
    l_lowy = max(leftEye[4:, 1])
    l_dify = abs(l_uppery - l_lowy)

    # gözün genişliğini hesapla
    lw = (leftEye[3][0] - leftEye[0][0])

    # CNN modeli için görüntünün (26, 34) olmasını istiyoruz
    # bu yüzden x ve y noktalarındaki farkın yarısını topladık
    # genişlikten yükseklik sırasıyla sol-sağ ve yukarı-aşağı ekseni

    minxl = (leftEye[0][0] - ((34 - lw) / 2))
    maxxl = (leftEye[3][0] + ((34 - lw) / 2))
    minyl = (l_uppery - ((26 - l_dify) / 2))
    maxyl = (l_lowy + ((26 - l_dify) / 2))

    # kareden göz kısmını kırp
    left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
    left_eye_rect = left_eye_rect.astype(int)
    left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]

    # yukardaki işlemleri sağ göz için de yapıyoruz
    r_uppery = min(rightEye[1:3, 1])
    r_lowy = max(rightEye[4:, 1])
    r_dify = abs(r_uppery - r_lowy)
    rw = (rightEye[3][0] - rightEye[0][0])
    minxr = (rightEye[0][0] - ((34 - rw) / 2))
    maxxr = (rightEye[3][0] + ((34 - rw) / 2))
    minyr = (r_uppery - ((26 - r_dify) / 2))
    maxyr = (r_lowy + ((26 - r_dify) / 2))
    right_eye_rect = np.rint([minxr, minyr, maxxr, maxyr])
    right_eye_rect = right_eye_rect.astype(int)
    right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

    # sol veya sağ gözü algılamazsa boş değeri döndür
    if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
        return None
    # CNN modeli için resimleri yeniden boyutlandırma
    left_eye_image = cv2.resize(left_eye_image, (34, 26))
    right_eye_image = cv2.resize(right_eye_image, (34, 26))

    # sağ ve sol göz için ayrı bir veri tabanı olmadığı için
    # sağ gözü sol gözle aynı yöne çevirip yolluyoruz
    right_eye_image = cv2.flip(right_eye_image, 1)

    # sol gözü ve gözü geri döndür
    return left_eye_image, right_eye_image


# fotoğrafı modelle uyumlu çözünürlüğe ayarlıyoruz formata çevirme
def cnnPreprocess(img):
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    return img


def main():
    global xTop, yTop, fare_x, fare_y

    t = None
    w = None

    # fare indexleri
    index_x = 322
    index_y = 360
    
    cap = cv2.VideoCapture(0)
    font_type = cv2.FONT_HERSHEY_DUPLEX

    kalibre = False
    cift_kirp_say = 0
    ilk_tarih = 0
    state_l = ''
    state_r = ''

    model = load_model('kaynak\kirpma_model.hdf5')  # CNN model dosyası çağrılıyor

    ekran_x, ekran_y = pyautogui.size()
    print(ekran_x, ekran_y)

    while True:
        ret, img = cap.read()

        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        # Fare tıklamaları
        if kalibre:
            t.penup()

            eyes = crop_eyes(img)  # görüntüden gözleri kırpma fonksiyonu
            if eyes is None:
                continue
            else:
                left_eye, right_eye = eyes
            # Kesilmiş iki göz resmi modele yüklenerek tahmin edilir
            prediction_r = (model.predict(cnnPreprocess(right_eye)))
            prediction_l = (model.predict(cnnPreprocess(left_eye)))

            if prediction_r < 0.1:  # tahminden dönen değer 0.1 değerinden küçükse göz kırpma olduğunu belirtir
                if prediction_l < 0.3:
                    cift_kirp_say += 1
                    if cift_kirp_say == 1:
                        ilk_tarih = datetime.datetime.now()
                    if cift_kirp_say < 4:
                        cv2.putText(img, "Cift goz kirpma", (200, 60), font_type, 0.7, (0, 0, 255), 1)
                        pyautogui.click()  # fare sol tiklaması
                        w.bgcolor("purple")
                        time.sleep(0.2)
                    if cift_kirp_say == 3:
                        cift_kirp_say = 0
                        son_tarih = datetime.datetime.now()
                        # print(son_tarih.second - ilk_tarih.second, "Saniye")
                        if son_tarih.second - ilk_tarih.second < 3:
                            kalibre = False
                            print("Tekrar Kalibre Ediliyor")

            if prediction_r < 0.01:
                if prediction_l > 0.5:
                    state_r = 'Kapali'
                    pyautogui.click(button='right')  # fare sağ tiklamas
                    w.bgcolor("red")
                    time.sleep(0.2)
            else:
                state_r = 'Acik'

            if prediction_l < 0.3:
                if prediction_r > 0.1:
                    state_l = 'Kapali'
                    pyautogui.click(clicks=2)  # fare çift tıklaması
                    w.bgcolor("blue")
                    time.sleep(0.2)
            else:
                state_l = 'Acik'

        for rect in rects:
            cv2.rectangle(img, (275, 230), (400, 255), (0, 0, 0), 1)

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # daha kesin bir göz kısmı çıkarmak için dlib kullanılıyoruz
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask = eye_on_mask(shape, mask, left)
            mask = eye_on_mask(shape, mask, right)
            eyes = cv2.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [0, 255, 255]
            mid = (shape[42][0] + shape[39][0]) // 2
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

            # gözün tespit edilmesi için bir kararma değeri veriyoruz
            _, thresh = cv2.threshold(eyes_gray, 100, 255, cv2.THRESH_BINARY)

            # gözlerin daha belirgin olması için fotoğraf filtreleme işlemleri
            thresh = cv2.bitwise_not(thresh)

            #sağ ve sol göz'ün etrafı ayrı ayrı çiziliyor
            contouring(thresh[:, 0:mid], mid, img)
            contouring(thresh[:, mid:], mid, img, True)

            xTop = shape[67][0]
            yTop = shape[67][1]

            # fare imleç kontrolü
            if kalibre:
                w.bgcolor("black")
                t.fillcolor("white")
                turtle_x = -5
                turtle_y = 0

                kose_x = 0
                kose_y = 0

                imlec_konumla = False
                if xTop - index_x > 6:
                    fare_x += xTop - index_x - 6
                    imlec_konumla = True

                    turtle_x += 20
                    t.shape("arrow")
                    t.shapesize(0.7)
                    t.goto(turtle_x, turtle_y)
                    kose_x = 2

                if xTop - index_x < - 6:
                    fare_x += xTop - index_x - 6
                    imlec_konumla = True
                    turtle_x -= 20
                    t.shape("arrow")
                    t.shapesize(0.7)
                    t.goto(turtle_x, turtle_y)
                    kose_x = 1

                if yTop - index_y > 6:
                    imlec_konumla = True
                    fare_y += yTop - index_y - 6
                    turtle_y -= 20
                    t.shape("arrow")
                    t.shapesize(0.7)
                    t.goto(turtle_x, turtle_y)
                    kose_y = 1

                if yTop - index_y < -6:
                    imlec_konumla = True
                    fare_y += yTop - index_y - 6
                    turtle_y += 20
                    t.shape("arrow")
                    t.shapesize(0.7)
                    t.goto(turtle_x, turtle_y)
                    kose_y = 2

                if kose_x == 1 and kose_y == 1:
                    t.tiltangle(225)
                elif kose_x == 2 and kose_y == 2:
                    t.tiltangle(45)
                elif kose_x == 1 and kose_y == 2:
                    t.tiltangle(135)
                elif kose_x == 2 and kose_y == 1:
                    t.tiltangle(315)
                elif kose_x == 1 and kose_y == 0:
                    t.tiltangle(180)
                elif kose_x == 2 and kose_y == 0:
                    t.tiltangle(360)
                elif kose_x == 0 and kose_y == 2:
                    t.tiltangle(90)
                elif kose_x == 0 and kose_y == 1:
                    t.tiltangle(270)

                if imlec_konumla:  # imleci son verilmiş konuma götür
                    pyautogui.moveTo(fare_x, fare_y)
                elif yTop - index_y < 8:
                    if yTop - index_y > -5:
                        t.shape("circle")
                        t.shapesize(0.9)
                        t.goto(-5, 0)

                fare_durum_x, fare_durum_y = pyautogui.position()
                if fare_durum_x > ekran_x - 150:
                    if fare_durum_y > ekran_y - 180:
                        w.setup(width=100, height=120, startx=ekran_x - 250, starty=ekran_y - 300)
                    else:
                        w.setup(width=100, height=120, startx=ekran_x - 130, starty=ekran_y - 180)
                else:
                    w.setup(width=100, height=120, startx=ekran_x - 130, starty=ekran_y - 180)

            # kalibrasyon
            else:
                if 330 > xTop > 305 and 350 > yTop > 330:

                    cv2.destroyAllWindows()  # pencereleri kapat
                    kalibre = True  # kalibre edildiği için diğer dönüşte koşula girmeyecek
                    print("Kalibre Edildi ")

                    fare_x = ekran_x / 2
                    fare_y = ekran_y / 2
                    pyautogui.moveTo(fare_x, fare_y)  # fareyi merkeze al

                    # turtle ekranını başlat ve ayarla
                    t = turtle.Turtle()
                    t.penup()
                    w = Screen()
                    w.resetscreen()
                    w.setup(width=100, height=120, startx=ekran_x - 130, starty=ekran_y - 180)
                    t.goto(-5, 0)
                    t.speed(0)
                # kalibrasyondan önce
                else:
                    cv2.putText(img, "Kalibrasyon icin ", (150, 90), font_type, 0.9, (0, 0, 255), 2)
                    cv2.putText(img, "Gozlerinizi ortada tutun ", (120, 115), font_type, 0.9, (0, 0, 255), 2)

                    # göz durumunu ekrana yazdır
                    cv2.putText(img, "Durum sag: {}".format(state_r), (300, 30), font_type, 0.5, (0, 0, 255), 1)
                    cv2.putText(img, "Durum sol: {}".format(state_l), (300, 60), font_type, 0.5, (0, 0, 255), 1)

                    cv2.imshow('Goz', img)  # kalibrasyon edilene kadar ekranda yüzü göster

        cv2.waitKey(1)

        if keyboard.is_pressed('q'):  # q basılırsa
            print('Çıkış')
            break  # döngüyü kır

    # kamerayı kapat
    cap.release()


if __name__ == '__main__':
    main()
