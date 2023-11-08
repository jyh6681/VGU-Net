import os
import pydicom as dicom
import cv2
import multiprocessing
import numpy
import nibabel



PATIENT_DICOM_FILEDIR_PATH = "caichuansong"
PATIENT_NII_FILE_PATH = "Brats18_2013_2_1_t1ce.nii"

IMG_DATA_DIR_PATH = "Image"
IMG_MASK_DATA_DIR_PATH = "Mask"
IMG_NODULE_AREA_DATA_DIR_PATH = "GetNodule"
IMG_DREW_DATA_DIR_PATH = "DrawEdge"

IMG_SAVED_TYPE = ".bmp"

def ImageGrayTranformation(ImgArrayInput, ImgNewMin = 0, ImgNewMax = 255):
    ImgArrOldMax = ImgArrayInput.max()
    ImgArrOldMin = ImgArrayInput.min()

    ImgArrNewMax = ImgNewMax
    ImgArrNewMin = ImgNewMin

    ImgArrayOutput = ((ImgArrNewMax - ImgArrNewMin) / (ImgArrOldMax - ImgArrOldMin)) * (ImgArrayInput - ImgArrOldMin) + ImgArrNewMin
    return ImgArrayOutput

"""Draw Nodule Edge"""
def DrawEdgeInImgFromMask(ImgDataDirPathInput, ImgMaskDataDirPathInput, ImgDstOutputDataDirPathOutput):
    assert os.path.exists(ImgDataDirPathInput)
    assert os.path.exists(ImgMaskDataDirPathInput)

    if not os.path.exists(ImgDstOutputDataDirPathOutput):
        os.makedirs(ImgDstOutputDataDirPathOutput)
    ImgDataFileNameList = os.listdir(ImgMaskDataDirPathInput)
    ImgDataFileNameListNumAll = len(ImgDataFileNameList)
    ImgDataFileNameListCount = 0
    for ImgDataFileNameCur in ImgDataFileNameList:
        ImgDataGray = cv2.imread(os.path.join(ImgMaskDataDirPathInput, ImgDataFileNameCur), cv2.COLOR_BGR2GRAY)
        #ret, binary = cv2.threshold(ImgGray,127,255,cv2.THRESH_BINARY)
        ImgGrayTemp, Contours, Hierarchy = cv2.findContours(ImgDataGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ImgDataWillDrawDst = cv2.imread(os.path.join(ImgDataDirPathInput, ImgDataFileNameCur))
        cv2.drawContours(ImgDataWillDrawDst, Contours, -1, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(ImgDstOutputDataDirPathOutput, ImgDataFileNameCur),ImgDataWillDrawDst)
        ImgDataFileNameListCount += 1
        print(os.path.join(ImgDstOutputDataDirPathOutput, ImgDataFileNameCur) + " " + str(ImgDataFileNameListNumAll - ImgDataFileNameListCount).zfill(6))


"""Get Nodule Area"""
def GetSickAreaInImgFromMask(ImgDataDirPathInput, ImgMaskDataDirPathInput, ImgDstOutputDataDirPathOutput):
    assert os.path.exists(ImgDataDirPathInput)
    assert os.path.exists(ImgMaskDataDirPathInput)

    if not os.path.exists(ImgDstOutputDataDirPathOutput):
        os.makedirs(ImgDstOutputDataDirPathOutput)
    ImgDataFileNameList = os.listdir(ImgMaskDataDirPathInput)
    ImgDataFileNameListNumAll = len(ImgDataFileNameList)
    ImgDataFileNameListCount = 0
    for ImgDataFileNameCur in ImgDataFileNameList:
        ImgDataGray = cv2.imread(os.path.join(ImgMaskDataDirPathInput, ImgDataFileNameCur), cv2.COLOR_BGR2GRAY)
        ImgDataWillGetNoduleAreaDst = cv2.imread(os.path.join(ImgDataDirPathInput, ImgDataFileNameCur), cv2.COLOR_BGR2GRAY)
        ImgDataWillGetNoduleAreaDst[ImgDataGray == 0] = 0
        cv2.imwrite(os.path.join(ImgDstOutputDataDirPathOutput, ImgDataFileNameCur), ImgDataWillGetNoduleAreaDst)
        ImgDataFileNameListCount += 1
        print(os.path.join(ImgDstOutputDataDirPathOutput, ImgDataFileNameCur) + " " + str(ImgDataFileNameListNumAll - ImgDataFileNameListCount).zfill(6))

"""Get Image Mask From Nii"""
def GenerateImageMask(NiiFileNameInput, ImgSavedDirAbsInput, ImgType = IMG_SAVED_TYPE):
    assert os.path.exists(NiiFileNameInput)

    if not os.path.exists(ImgSavedDirAbsInput):
        os.makedirs(ImgSavedDirAbsInput)
    #Get Nii Data
    ImgNii = nibabel.load(NiiFileNameInput)
    #Get Pixel Data (Matrix)
    ImgNiiMatrixData = ImgNii.get_data()
    #Get Img 
    ImgNiiNumAll = ImgNiiMatrixData.shape[2]
    for ImgNumCur in range(ImgNiiNumAll):
        #ImgFileNameCur = str(ImgNumCur).zfill(6) + ImgType
        ImgFileNameCur = str(ImgNiiNumAll - ImgNumCur).zfill(6) + ImgType
        ImgNiiMatrixDataSingle = ImgNiiMatrixData[:,:,ImgNumCur]
        #ImgNiiMatrixDataSingle[ImgNiiMatrixDataSingle != 0 ] = 255
        ImgNiiMatrixDataSingle = ImgNiiMatrixDataSingle.T
        #Tranform
        ImgNiiMatrixDataSingle = ImageGrayTranformation(ImgNiiMatrixDataSingle)
        print("ImgNiiMatrixDataSingle.max():" + str(ImgNiiMatrixDataSingle.max()))
        print("ImgNiiMatrixDataSingle.min():" + str(ImgNiiMatrixDataSingle.min()))
        cv2.imwrite(os.path.join(ImgSavedDirAbsInput, ImgFileNameCur), ImgNiiMatrixDataSingle)
        print("Processing: " + ImgFileNameCur + " " + str(ImgNiiNumAll - ImgNumCur).zfill(6))



"""加窗显示技术"""
def WindowsDisplayTechnology(DicomSingleData):
    SingleDicomArray = DicomSingleData.pixel_array.copy()

    """Hu值计算"""
    Intercept = DicomSingleData.RescaleIntercept
    Slope = DicomSingleData.RescaleSlope
    '''
    if Slope != 1:  # 按照Hu计算的公式	当sloce不为1的时候 则按照公式计算
        SingleDicomArray = Slope * SingleDicomArray.astype(numpy.float64)  # 先乘截距
        SingleDicomArray = SingleDicomArray.astype(numpy.int16)  # 再转换数据类型
        SingleDicomArray += numpy.int16(Intercept)
    '''
    if True:  # 按照Hu计算的公式	当sloce不为1的时候 则按照公式计算
        SingleDicomArray = Slope * SingleDicomArray.astype(numpy.float64)  # 先 乘截距
        SingleDicomArray = SingleDicomArray.astype(numpy.int16)  # 再转换数据类型
        SingleDicomArray += numpy.int16(Intercept)

    '''提取 Window_Center AND Window_Center'''
    try:
        Window_Center = DicomSingleData.WindowCenter  # 窗位
        Window_Width = DicomSingleData.WindowWidth  # 窗宽
        print("Window_Center:" + str(Window_Center))
        print("Window_Width:" + str(Window_Width))
    except AttributeError:
        print("AttributeError  Window_Center Window_Width")
        Window_Center = -650  # 窗位
        Window_Width = 1600  # 窗宽
        print("Window_Center:" + str(Window_Center))
        print("Window_Width:" + str(Window_Width))

    '''不是正常INT类型,进行强制类型转换'''
    if int != type(Window_Center):
        print("Window_Center 不是INT类型, 将进行强制更改!!!")
        Window_Center = -650
        print("Window_Center:" + str(Window_Center))
    if int != type(Window_Width):
        print("Window_Width 不是INT类型, 将进行强制更改!!!")
        Window_Width = 1600
        print("Window_Width:" + str(Window_Width))


    '''如果不是在正常肺窗范围之内，则进行强制转换'''
    if Window_Center > -400 or Window_Center < -700:
        print("Window_Center 不是正常范围, 将进行强制更改!!!")
        Window_Center = -650
        print("Window_Center:" + str(Window_Center))
    if Window_Width > 2000 or Window_Width < 1000:
        print("Window_Width 不是正常范围, 将进行强制更改!!!")
        Window_Width = 1600
        print("Window_Width:" + str(Window_Width))

    '''加窗显示技术'''
    On_the_left_of_Window_Center = Window_Center - numpy.int16(Window_Width / 2.0)
    On_the_right_of_Window_Center = Window_Center + numpy.int16(Window_Width / 2.0)
    print("On_the_left_of_Window_Center:" + str(On_the_left_of_Window_Center))
    print("On_the_right_of_Window_Center:" + str(On_the_right_of_Window_Center))

    Thresshold_left = numpy.int16(
        ((On_the_left_of_Window_Center - (Window_Center - 0.5)) / (Window_Width - 1) + 0.5) * 255.0)
    Thresshold_right = numpy.int16(
        ((On_the_right_of_Window_Center - (Window_Center - 0.5)) / (Window_Width - 1) + 0.5) * 255.0)
    print("Thresshold_left:" + str(Thresshold_left))
    print("Thresshold_right:" + str(Thresshold_right))

    '''加窗显示技术公式计算'''
    SingleDicomArray = numpy.float64(SingleDicomArray)
    print("SingleDicomArray.max():" + str(SingleDicomArray.max()))
    print("SingleDicomArray.min():" + str(SingleDicomArray.min()))

    SingleDicomArray = ((SingleDicomArray - (Window_Center - 0.5)) / (Window_Width - 1) + 0.5) * 255.0  # 加窗显示技术

    SingleDicomArray = numpy.int16(SingleDicomArray)
    print("SingleDicomArray.max():" + str(SingleDicomArray.max()))
    print("SingleDicomArray.min():" + str(SingleDicomArray.min()))

    SingleDicomArray[SingleDicomArray < Thresshold_left] = 0
    SingleDicomArray[SingleDicomArray > Thresshold_right] = 255

    print("SingleDicomArray.max():" + str(SingleDicomArray.max()))
    print("SingleDicomArray.min():" + str(SingleDicomArray.min()))

    return SingleDicomArray

"""Get Image From Dicom"""
def GetImageFromDicom(ImgDicomDirPathInput, ImgSavedDirAbsInput, ImgType = IMG_SAVED_TYPE):
    assert os.path.exists(ImgDicomDirPathInput)

    if not os.path.exists(ImgSavedDirAbsInput):
        os.makedirs(ImgSavedDirAbsInput)
    ImgFileNameAll = os.listdir(ImgDicomDirPathInput)
    ImgDicomFileNameAll = []
    for ImgFileNameSingle in ImgFileNameAll:
        if ".dcm" in ImgFileNameSingle:
            ImgDicomFileNameAll.append(ImgFileNameSingle)

    ImgDicomDataAll = [dicom.read_file(os.path.join(ImgDicomDirPathInput, ImgDicomFileNameSingle)) for ImgDicomFileNameSingle in ImgDicomFileNameAll]
    
    ImgDicomDataAll.sort(key=lambda x:x.InstanceNumber)

    ImgCount = 0
    for ImgDicomDataSingle in ImgDicomDataAll:
        ImgMatrixData = WindowsDisplayTechnology(ImgDicomDataSingle)
        ImgCount += 1
        ImgFileNameCur = str(ImgCount).zfill(6) + ImgType
        cv2.imwrite(os.path.join(ImgSavedDirAbsInput, ImgFileNameCur), ImgMatrixData)
        print(os.path.join(ImgSavedDirAbsInput, ImgFileNameCur) + " " + str(len(ImgDicomFileNameAll) - ImgCount).zfill(6))



if __name__ == '__main__':
    #GetImageFromDicom(PATIENT_DICOM_FILEDIR_PATH, IMG_DATA_DIR_PATH)
    GenerateImageMask(PATIENT_NII_FILE_PATH, IMG_MASK_DATA_DIR_PATH)
    #GetSickAreaInImgFromMask(IMG_DATA_DIR_PATH, IMG_MASK_DATA_DIR_PATH, IMG_NODULE_AREA_DATA_DIR_PATH)
    #DrawEdgeInImgFromMask(IMG_DATA_DIR_PATH, IMG_MASK_DATA_DIR_PATH, IMG_DREW_DATA_DIR_PATH)


