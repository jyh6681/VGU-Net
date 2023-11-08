import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import spectral_clustering,DBSCAN,SpectralClustering
# input a image path and a numpy array heatmap
from scipy.ndimage import measurements
from numba import jit
@jit(nopython=True)
def find_max(objects,num):
    W, H = np.shape(objects)
    value_list = []
    for w in range(0, W):
        for h in range(0, H):
                #print("value:",objects[w, h, d])
                if objects[w, h] != 0:
                    tmp = objects[w, h]
                    value_list.append(tmp)
    sum_list = np.zeros(num + 1)
    for j in range(1, num + 1):
        sum = 0
        for t in value_list:
            value = t
            if value == j:
                sum = sum + 1
        sum_list[j] = sum
    return sum_list

def get_max_region(image: np.ndarray, for_which_classes=[1.0]):
    image=image/255
    assert 0 not in for_which_classes, "cannot remove background"
    out = image
    s = np.ones((3,3))
    for c in [0.0,1.0]:
        mask = np.zeros_like(image)
        mask[image == c] = 1
        objects, num = measurements.label(mask, structure=s)  ###numpy with entries 1,2,3,4
        print("Class", c, "has ", num, "connected components")
        ##sparse way
        sum_list = find_max(objects,num)
        max_pixel = np.argmax(sum_list)
        out[image==c]=0
        if c==2:
            first_max = np.argmax(sum_list)
            sum_list[first_max]=0
            second_max = np.argmax(sum_list)
            out[objects==first_max]=c
            out[objects==second_max]=c
        else:
            out[objects == max_pixel] = c
    return out




single_slice=40
scale=2
all_adj = np.load("./output/msugnet/plt_"+str(single_slice)+"/adjmap"+str(scale)+"_brats.npy")
# print("adj:",all_adj[50:160,40].tolist())
# all_adj[all_adj<0.001]=0
up=0 ##keep uppertrianngle matrix to form sysmentric adjacency
if up==1:
    upper=np.triu(all_adj,k=0)
    lower=np.transpose(np.triu(all_adj,k=1))
    all_adj = upper+lower

N=int(160/scale)
slice_list=[100]
for slice in slice_list:
    width = int(160/scale)
    adj = all_adj[slice,:].reshape((width,width))
    print("origin:",adj.max())
    att_map = adj.repeat(scale, axis = 0).repeat(scale, axis = 1)/adj.max()
    min = att_map.min()
    att_map = (att_map-min)/(1-min)
    # print("show:",att_map[0,:].tolist())
    if single_slice == 54:
        img = cv2.imread("./output/msugnet/plt_54/gt_BraTS19_CBICA_BGE_1_71.png")
        class_num = 4
    if single_slice==26:
        img = cv2.imread("./output/msugnet/plt_26/gt_BraTS19_CBICA_AYC_1_50.png")
        class_num = 4
    if single_slice==40:
        img = cv2.imread("./output/msugnet/plt_40/gt_BraTS19_CBICA_BGG_1_93.png")
        class_num = 2
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


    y =  (slice//width)*scale
    x = scale*(slice-width*(slice//width))+scale
    pt1 = (x, y)  # 左边，上边   #数1 ， 数2
    pt2 = (x + scale, y + scale)  # 右边，下边  #数1+数3，数2+数4
    rimg = img.copy()
    # rimg = cv2.rectangle(rimg, pt1, pt2, (255, 0, 0), thickness=1)####red box to index
    # cv2.imwrite(str(slice)+"attmap.jpg",rimg)


    ##clustering methods
    cluster=True
    if cluster==True:
        # adj = spectral_clustering(all_adj,n_clusters=class_num,n_init=20)
        spec = SpectralClustering(n_clusters=class_num,n_init=50,affinity="precomputed")
        spec.fit(all_adj)
        adj = spec.fit_predict(all_adj)
        print("out:",adj)
        adj = 255*(adj/(class_num-1))
        adj = adj.reshape((N,N)) #####reshape how
        # adj = get_max_region(adj)
        print("max:",adj.max(),adj)
        plt.imshow(adj)
        plt.savefig("./output/msugnet/plt_" + str(single_slice) + "/directshow" + str(scale) + "_brats.png")
        plt.show()
        att_map = adj.repeat(scale, axis = 0).repeat(scale, axis = 1)/adj.max()

    fig,axes = plt.subplots(nrows=1, ncols=2)
    for ax in axes:
        ax.axis('off')
    plt.axis('off')
    axes[0].imshow(rimg)
    plt.axis('off')
    # plt.tight_layout(w_pad=1)

    im = axes[1].imshow(np.zeros_like(img))
    att = axes[1].imshow(att_map, alpha=1, cmap="jet")
    # plt.colorbar(att,ax=axes,fraction=0.02)
    #plt.tight_layout(w_pad=1)
    plt.savefig("./output/msugnet/plt_"+str(single_slice)+"/segshow"+str(scale)+"_brats.png")
    # plt.show()


# hm = HeatMap(img,att_map,gaussian_std=0,)
# hm.plot(transparency=0.5,color_map="jet",show_original=True,show_colorbar=True)

# 画矩形框 距离靠左靠上的位置

# a = 'people'  # 类别名称
# b = 0.596  # 置信度
# font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
# imgzi = cv2.putText(img, '{} {:.3f}'.format(a, b), (651, 460 - 15), font, 1, (0, 255, 255), 4)

