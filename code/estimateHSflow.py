import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.sparse.linalg as sslg
import scipy.sparse as ss
import flowTools as ft


def estimateHSflowlayer(frame1, frame2, uv, lam=80, maxwarping=10):
    H, W = frame1.shape

    npixels = H * W

    x, y = np.meshgrid(range(W), range(H))

    # TODO#3: build differential matrix and Laplacian matrix according to
    # image size
    xx = np.ones([2,npixels])
    xx[0,:]=-xx[0,:]
    Dx = ss.spdiags(xx,[0,1],npixels,npixels,format = 'csc')
    Dy = ss.spdiags(xx,[0,H],npixels,npixels,format = 'csc')
    L = Dx.transpose() * Dx + Dy.transpose() * Dy


    # please use order 'F' when doing reshape: np.reshape(xx, xxshape, order='F') 

    # Kernel to get gradient
    h = np.array( [[1, -8, 0, 8, -1]], dtype='single' ) / 12

    for i in range(maxwarping):

        # TODO#2: warp image using the flow vector
        # an example is in runFlow.py
        remap = np.zeros([H, W, 2])
        remap[:, :, 0] = x + uv[:, :, 0]
        remap[:, :, 1] = y + uv[:, :, 1]
        remap = remap.astype('single')
        warped2 = cv2.remap(frame2, remap, None, cv2.INTER_CUBIC)  
        warped2[warped2 == np.nan] = 0
        
        # TODO#4: compute image gradient Ix, Iy, and Iz
        Ix = cv2.filter2D( warped2, -1, h )
        Iy = cv2.filter2D( warped2, -1, h.transpose() )
        Iz = warped2 - frame1
        Ix = np.reshape(Ix, [1,npixels], order='F')
        Iy = np.reshape(Iy, [1,npixels], order='F')
        Iz = np.reshape(Iz, [1,npixels], order='F')
        U = np.reshape(uv[:, :, 0], [1,npixels], order='F')
        V = np.reshape(uv[:, :, 1], [1,npixels], order='F')
        Ix = ss.spdiags(Ix,[0],npixels,npixels,format = 'csc')
        Iy = ss.spdiags(Iy,[0],npixels,npixels,format = 'csc')

        # TODO#5: build linear system to solve HS flow
        # generate A,b for linear equation Ax = b
        # you may need use scipy.sparse.spdiags
        A11 = Ix*Ix + lam*L
        A12 = Ix*Iy
        A21 = A12
        A22 = Iy*Iy + lam*L
        A = ss.vstack( [ss.hstack([A11,A12]), ss.hstack([A21,A22])] ,format = 'csc')

        b1 = Ix.dot(Iz.transpose()) + lam*L.dot(U.transpose())
        b2 = Iy.dot(Iz.transpose()) + lam*L.dot(V.transpose())
        b = -np.vstack( [b1, b2] )

        ret = sslg.spsolve(A, b)
        deltauv = np.reshape(ret, uv.shape, order='F')

        deltauv[deltauv is np.nan] = 0
        deltauv[deltauv > 1] = 1
        deltauv[deltauv < -1] = -1

        uv = uv + deltauv
        uv[:, :, 0] = cv2.medianBlur(uv[:, :, 0].astype('single'),5) 
        uv[:, :, 1] = cv2.medianBlur(uv[:, :, 1].astype('single'),5) 
        print 'Warping step: %d, Incremental norm: %3.5f' %(i, np.linalg.norm(deltauv))
        # Output flow

    return uv


def estimateHSflow(frame1, frame2, lam = 80):
    H, W = frame1.shape

    # build the image pyramid
    pyramid_spacing = 1.0 / 0.8
    pyramid_levels = 1 + np.floor(np.log(min(W, H) / 16.0) / np.log(pyramid_spacing * 1.0))
    smooth_sigma = np.sqrt(2.0)

    pyramid1 = []
    pyramid2 = []

    pyramid1.append(frame1)
    pyramid2.append(frame2)
    
    pyramid_levels = pyramid_levels.astype('int')
    for m in range(1, pyramid_levels):
        # TODO #1: build Gaussian pyramid for coarse-to-fine optical flow
        # estimation
        # use cv2.GaussianBlur
        H1, W1 = pyramid1[m-1].shape
        H2 = int(H1 / pyramid_spacing)
        W2 = int(W1 / pyramid_spacing)
        #print 'H %d, W %d' %(H2,W2)
        framelayer1 = cv2.GaussianBlur(pyramid1[m-1],(5,5),smooth_sigma)  
        framelayer1 = cv2.resize(framelayer1, (W2, H2))
        framelayer2 = cv2.GaussianBlur(pyramid2[m-1],(5,5),smooth_sigma)  
        framelayer2 = cv2.resize(framelayer2, (W2, H2))
        pyramid1.append(framelayer1)
        pyramid2.append(framelayer2)
    # coarst-to-fine compute the flow
    uv = np.zeros(((H, W, 2)))

    for levels in range(pyramid_levels - 1, -1, -1):
        print "level %d" % (levels)
        H1, W1 = pyramid1[levels].shape
        uv = cv2.resize(uv, (W1, H1))
        uv = estimateHSflowlayer(pyramid1[levels], pyramid2[levels], uv, lam,3)

        # TODO #6: use median filter to smooth the flow result in each level in each iteration

        uv[:, :, 0] = cv2.medianBlur(uv[:, :, 0].astype('single'),5) 
        uv[:, :, 1] = cv2.medianBlur(uv[:, :, 1].astype('single'),5) 

    return uv