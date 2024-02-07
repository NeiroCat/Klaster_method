import os.path
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from scipy import signal
from mpl_toolkits import mplot3d
#from scipy import datasets
# specify your image path
#dicom_dir = r'C:\Users\1\Desktop\работа\d2g3rat2'
#export_location = r'C:\Users\1\Desktop\работа\сюда'
#image_path = r'C:\Users\1\Desktop\работа\d2g3rat2\d2g3rat2_0050.dcm'
#фукнкция для работы сразу со всеми файлами папки (определение имени файла)

value = False
paints = False
txt = False
true_paints = False
contrast_up = False
NN_value = False
gistogramm = True

directory = 'test_data/' #ЗАМЕНИТЬ НАЗВАНИЕ ПАПКИ ДЛЯ НОВОГО ЭКСПЕРИМЕНТА
output_path = '2023rats/jpg/4G_25_10/4G_rat1/4G_rat1_3week_15_11/'
gistogramm_name = '4G_rat1_3week_15_11'
area = False
NN_directory_img = 'rats/RATS_G3/G3 Rat2/d16g3rat2 week16/'
NN_directory_mask = 'для теста/NN rat 3.2/3.2/output/d16g3rat2 week16/'
def get_names(path):
    names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.dcm']:
                names.append(filename)
    return names

def get_names_NN(path):
    names_NN = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.jpg']:
                names_NN.append(filename)
    return names_NN

names = get_names(directory)

# переводим в jpg
def convert_dcm_jpg(name):
    ds = dicom.dcmread(directory + name) #ЗАМЕНИТЬ НАЗВАНИЕ ПАПКИ ДЛЯ НОВОГО ЭКСПЕРИМЕНТА, ДОБАВИТЬ ПАПКУ В ПРОЕКТ
    im = ds.pixel_array.astype(float)
    #im_invert = im/im.max()*255
    #im_invert = np.abs(im)
    im_invert = (np.maximum(np.abs(im), 0)/np.abs(im).max())*255
    #im_invert = (np.maximum(-im, 0) / im.max()) * 255
    final_image = np.uint8(im_invert)
    final_image = Image.fromarray(final_image)
    #final_image.save('new_image.png')
    return final_image

if true_paints is True:
    for name in names:
        image = convert_dcm_jpg(name)
        if contrast_up is True:
            enhancer = ImageEnhance.Contrast(image)
            contrast_image = enhancer.enhance(0.5)
            contrast_image.save(output_path + name + '.jpg')
        else:
            image.save(output_path+name+'.jpg') #добавить +директори+

#вывод полной матрицы для анализа
def matrix_analysis(name):
    ds = dicom.dcmread(directory + name)  # ЗАМЕНИТЬ НАЗВАНИЕ ПАПКИ ДЛЯ НОВОГО ЭКСПЕРИМЕНТА, ДОБАВИТЬ ПАПКУ В ПРОЕКТ
    im = ds.pixel_array.astype(float)
    str_matrix = np.array2string(im)
    return str_matrix

if txt is True:
    for name in names:
        ds = dicom.dcmread(directory + name)  # ЗАМЕНИТЬ НАЗВАНИЕ ПАПКИ ДЛЯ НОВОГО ЭКСПЕРИМЕНТА, ДОБАВИТЬ ПАПКУ В ПРОЕКТ
        im = ds.pixel_array.astype(float)
        f = open('матрицы/' + name + '.txt', 'w') # +directory+
        i_max, j_max = im.shape   # количество строк # количество столбцов
        for i in range(0,i_max): #идем по строкам
            f.write('\n\n\n\n')
            for j in range(0, j_max): #идем по столбцам
                f.write(np.array2string(im[i,j]))
                f.write('  ')



#sigmentation
def sigmentation(name):
    ds = dicom.dcmread(directory + name)  # ЗАМЕНИТЬ НАЗВАНИЕ ПАПКИ ДЛЯ НОВОГО ЭКСПЕРИМЕНТА, ДОБАВИТЬ ПАПКУ В ПРОЕКТ
    im = ds.pixel_array.astype(float)
    i_max, j_max = im.shape   # количество строк # количество столбцов

    # Gaussian filter
    scharr = np.array([[-1 - 1j, 0 - 1j, +1 - 1j],
                       [-1 + 0j, 0 + 0j, +1 + 0j],
                       [-1 + 1j, 0 + 1j, +1 + 1j]])
    #Gausimage = signal.convolve2d(im, scharr, boundary='symm', mode='same')
    a = i_max/2 * 55/100
    b = j_max/2 * 60/100
    i_0 = i_max/2
    j_0 = j_max/2
#    Gausimage = im.filter(ImageFilter.GaussianBlur)
    sig_image = np.copy(im)
    for i in range(0, i_max):  # идем по строкам
        for j in range(0, j_max):  # идем по столбцам
            if not sig_image[i, j] in range(-800, -300):
                sig_image[i, j] = sig_image[i, j] - sig_image[i, j]
            if not (((i - i_0) ** 2) /(a ** 2) + ((j - j_0)**2)/(b ** 2) < 1):
                sig_image[i, j] = sig_image[i, j] - sig_image[i, j]        # отсечение эллипса(для сигментации)
#    sig_image = signal.convolve2d(sig_image, scharr)
    return sig_image

#переводим в жипег отсигментированное
def convert_sig_jpg(name):
    im = sigmentation(name)
    im_invert = (np.maximum(np.abs(im), 0) / np.abs(im).max()) * 255
    final_image = np.uint8(im_invert)
    final_image = Image.fromarray(final_image)
    return final_image


#image sigmentation
if paints is True:
    for name in names:
        image = convert_sig_jpg(name)
        image.save(output_path + name + '_mask.jpg') #+directory+


#определение мю легких
#def mu_loung_im(directory):
#    mu_loung =
#    return mu_loung


#пересчет в еденицах Хаусфилда
#def HU_convert(mu_water, mu_air, mu_loung):
#    HU_loung = 1000 * (mu_loung - mu_water)/(mu_water - mu_air)
#    return HU_loung

#Значения коэффициентов линейного ослабления
mu_air = -1000 #коэффициент линейного ослабления воздуха(заменить при необходимости)
mu_water = 0 #коэффициент линейного ослабления воды(заменить при необходимости)
#mu_loung = mu_loung_im('d2g3rat2') #ЗАМЕНИТЬ НАЗВАНИЕ ПАПКИ ДЛЯ НОВОГО ЭКСПЕРИМЕНТА


#среднее HU легких
def HU_loung(directory):
    Min = -1000
    Max = 500
    N = 50
    S = 0
    k = 0
    z = 0
    L = 0
    a = np.zeros((50, 500), dtype=int)
    b = np.zeros(50)
#    c = np.zeros(50)
    names = get_names(directory)  # ЗАМЕНИТЬ НАЗВАНИЕ ПАПКИ ДЛЯ НОВОГО ЭКСПЕРИМЕНТА
    for name in names:
        im = sigmentation(name)
        ds = dicom.dcmread(directory + name)  # ЗАМЕНИТЬ НАЗВАНИЕ ПАПКИ ДЛЯ НОВОГО ЭКСПЕРИМЕНТА, ДОБАВИТЬ ПАПКУ В ПРОЕКТ
        orig_im = ds.pixel_array.astype(float)
        i_max, j_max = im.shape   # количество строк # количество столбцов
        for i in range(0, i_max):  # идем по строкам
            for j in range(0, j_max):# идем по столбцам
                if not im[i, j] == 0:
                    S += im[i, j]
                    k += 1
                    if gistogramm is True:
                        L = int(((im[i,j] + (- Min)) // N))
                        a[L, z] += 1
                        b[L] += 1
 #               if gistogramm is True:
 #                   L = int(((orig_im[i, j] + (- Min)) // N))
  #                  c[L] += 1
                #print('done', L,'   ', a[L, 1, z],'  ', a[L,0,z])
        z += 1
        print (z)
    #mat = np.matrix(a)

    with open(gistogramm_name + '.txt', 'w') as testfile:
        for row in a:
            testfile.write(' '.join([str(с) for с in row]) + '\n')

    HU_sr = round(S/k)
    fig, ax = plt.subplots()
    ax.plot(b)
#    ax.plot(c)
    plt.show()
    fig = plt.figure()
    #ax_3d = Axes3D(fig)
    ax = fig.add_subplot(projection="3d")
    xv = np.linspace(0, 30, 30)
    yv = np.linspace(0, 303, 303)
    xgrid, ygrid = np.meshgrid(xv, yv)
    zgrid = a[xgrid.astype(int), ygrid.astype(int)]
    ax.plot_surface(xgrid, ygrid, zgrid)
    #ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$");
    plt.show()

    if gistogramm is True:
        return a
    else:
        return HU_sr

def NN_loung(NN_directory_img, NN_directory_mask):
    S = 0
    k = 0
    names = get_names(NN_directory_img)  # ЗАМЕНИТЬ НАЗВАНИЕ ПАПКИ ДЛЯ НОВОГО ЭКСПЕРИМЕНТА
    mask_names = get_names_NN(NN_directory_mask)
    print(names)
    print(mask_names)
    for NN_name in mask_names:
        name = (NN_name.split('.')[0] + '.dcm')
        ds = dicom.dcmread(NN_directory_img + name)  # ЗАМЕНИТЬ НАЗВАНИЕ ПАПКИ ДЛЯ НОВОГО ЭКСПЕРИМЕНТА, ДОБАВИТЬ ПАПКУ В ПРОЕКТ
        im = ds.pixel_array.astype(float)
        img = im.copy()
        NN_im = Image.open(NN_directory_mask + NN_name).convert('L')
        NN_img = np.asarray(NN_im)
        print(NN_name)
#        print(np.max(NN_img))
#        print((NN_img))
        if (NN_name.split('.')[0]) == (name.split('.')[0]):
            i_max = 256
            j_max = 256  # количество строк # количество столбцов
            a = i_max / 2 * 55 / 100
            b = j_max / 2 * 70 / 100
            print(i_max, j_max)
            for i in range(0, i_max):  # идем по строкам
                for j in range(0, j_max):  # идем по столбцам
                    #print (i, '  ', j)
                    #print(NN_img[i,j])
#                    if NN_img[i, j] > 150:
#                        #print(img[i,j])
                        if img[i,j] < -50:
                            if area is True:
                                if not (((i - i_max / 2) ** 2) / (a ** 2) + ((j - j_max / 2) ** 2) / (b ** 2) < 1): # отсечение эллипса(для сигментации)
                                    S += img[i, j]
                                    k += 1
                            else:
                                S += img[i, j]
                                k += 1
                        #print(img[i,j])
        else: print('NOT DONE')
    print('done', S, '    ', k, '    ')

    HU_sr = round(S / k)
    return HU_sr

if value is True:
    HU = HU_loung(directory)
    print('HU' + directory + ' =', HU)
if gistogramm is True:

#        xval = np.arange(0, 30, 1)
#        yval = np.arange(0, 10000, 100)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        xv = np.linspace(0, 30, 30)
        yv = np.linspace(0, 1000, 1000)
        xgrid,ygrid = np.meshgrid(xv,yv)
        a = HU_loung(directory)
        zgrid = a[xgrid, 1, ygrid]
        ax.plot_surface(xgrid, ygrid, zgrid)
        ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$");
        plt.show()
        #ax.set_zlim(0,400)


if NN_value is True:
    HU = NN_loung(NN_directory_img, NN_directory_mask)
    print('HU' + NN_directory_img + ' =', HU)

#ds = dicom.dcmread(directory + 'd4g1rat3_0000.dcm')  # ЗАМЕНИТЬ НАЗВАНИЕ ПАПКИ ДЛЯ НОВОГО ЭКСПЕРИМЕНТА, ДОБАВИТЬ ПАПКУ В ПРОЕКТ
#im = ds.pixel_array.astype(float)
#i_max, j_max = im.shape
#print(i_max, j_max)
