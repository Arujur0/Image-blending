from asyncio import wait_for
import cv2
import numpy as np
import scipy as sp
from align_target import align_target
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import inv

def compute_A(temp, h, w):
    x = 0
    A = lil_matrix((h*w, h*w))
    for i in range(h):
        for j in range(w):
            if temp[i][j]>0:
                if i  in [0, h-1] and j in [0, w-1]:
                    A[x, i*w + j] = 1
                else:
                    if j in [0, w-1]:
                        A[x,i * w + j] = 2
                        A[x,(i-1) * w + j] = -1
                        A[x,(i+1) * w + j] = -1  
                    elif i in [0, h-1]:
                        A[x,i * w + j] = 2
                        A[x,i * w + (j-1)] = -1
                        A[x,i * w + (j+1)] = -1
                    else:
                        A[x,i * w+ j] = 4
                        A[x,i * w+ (j+1)] = -1
                        A[x,(i+1)*w+ j] = -1
                        A[x,(i-1) * w+ j] = -1
                        A[x,i*w+ (j-1)] = -1
            else:
                A[x, i*w + j] = 1                      
            x+=1       
    return A


def compute_b(source, target, temp, h, w):
    b = np.zeros(h*w)
    x = 0
    for i in range(h):
        for j in range(w):
            if temp[i][j]>0:
                if  i  in [0, h-1] and j in [0, w-1]:
                    b[x] = 25
                else:
                    if j in {0, w-1}:
                        b[x] =  (2 * source[i,j]) - source[i-1, j] - source[i+1,j]
                    elif i in [0, h-1]:
                        b[x] =  (2 * source[i,j]) - source[i, j-1] - source[i,j+1]
                    elif temp[i-1][j] == 0 or temp[i+1][j] == 0 or temp[i][j+1] == 0 or temp[i][j-1] == 0:
                        b[x] = target[i][j] 
                    else:
                        b[x] = (4 * source[i,j]) - source[i-1, j] - source[i+1,j] - source[i, j-1] - source[i,j+1]
            else:
                b[x] = target[i][j]
            x+=1
    return b


def poisson_blend(img, target_img, target_mask):
    #source_image: image to be cloned
    #target_image: image to be cloned into
    #target_mask: mask of the target image

    h = int(img.shape[0])
    w = int(img.shape[1])
    A = compute_A(target_mask, h, w)
    b = compute_b(img, target_img, target_mask, h, w)
    print(A.shape)
    inv = sp.sparse.linalg.spsolve(A, b)
    lse = A.dot(inv) - b
    ls = sp.linalg.norm(lse, ord = 2)
    print("Least square error is : ", ls)
    inv = inv.reshape(h,w)
    for i in range(h):
        for j in range(w):
            if inv[i][j] <0:
                inv[i][j] = 0
            elif inv[i][j] > 255:
                inv[i][j] = inv[i][j] % 255
    
    return inv
if __name__ == '__main__':
    #read source and target images
    source_path = 'Poisson blending\source1.jpg'
    target_path = 'Poisson blending\\target.jpg'
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)
    #align target image
    im_source, mask = align_target(source_image, target_image)
    img = im_source
    ##poisson blend
    for i in range(3):
        img[:,:,i] = poisson_blend(im_source[:,:,i], target_image[:,:,i], mask)
    
    cv2.imwrite('poisson output\\final.jpg', img)
