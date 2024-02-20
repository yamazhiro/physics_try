import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d')

# 3D座標設定
ax.set_aspect('equal')
# ax.set_title('evalt')
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_zlim(-10,10)
ax.set_xlabel("x", size = 14, color = "r")
ax.set_ylabel("y", size = 14, color = "r")
ax.set_zlabel("z", size = 14, color = "r")

def a_to_b(a1,a2,a3):
    b1 = 2* 3.14 * (np.cross(a2,a3))/(sum(a1*(np.cross(a2,a3))))
    b2 = 2* 3.14 * (np.cross(a3,a1))/(sum(a2*(np.cross(a3,a1))))
    b3 = 2* 3.14 * (np.cross(a1,a2))/(sum(a3*(np.cross(a1,a2))))

    kousi_bx = [0 for i in range(N*N*N)]
    kousi_by = [0 for i in range(N*N*N)]
    kousi_bz = [0 for i in range(N*N*N)]

    for l in range(N):
        for k in range(N):
            for h in range(N):
                kousi_bx[N*N*l + N*k + h] = b1[0]*h + b2[0]*k + b3[0]*l
                kousi_by[N*N*l + N*k + h] = b1[1]*h + b2[1]*k + b3[1]*l
                kousi_bz[N*N*l + N*k + h] = b1[2]*h + b2[2]*k + b3[2]*l

    return kousi_bx, kousi_by, kousi_bz

def ab_rotate(a1,a2,a3,theta_degree=60,N=5,ac='yellow',bc='blue'):
    theta = theta_degree*np.pi/180
    
    a1 = np.array([a1[0]*np.cos(theta)-a1[1]*np.sin(theta),a1[0]*np.sin(theta)+a1[1]*np.cos(theta),a1[2]])
    a2 = np.array([a2[0]*np.cos(theta)-a2[1]*np.sin(theta),a2[0]*np.sin(theta)+a2[1]*np.cos(theta),a2[2]])
    a3 = np.array([a3[0]*np.cos(theta)-a3[1]*np.sin(theta),a3[0]*np.sin(theta)+a3[1]*np.cos(theta),a3[2]])

    kousi_ax = [0 for i in range(N*N*N)]
    kousi_ay = [0 for i in range(N*N*N)]
    kousi_az = [0 for i in range(N*N*N)]

    for i in range(N):
        for j in range(N):
            for k in range(N):
                kousi_ax[N*N*i + N*j + k] = a1[0]*k + a2[0]*j + a3[0]*i
                kousi_ay[N*N*i + N*j + k] = a1[1]*k + a2[1]*j + a3[1]*i
                kousi_az[N*N*i + N*j + k] = a1[2]*k + a2[2]*j + a3[2]*i

    #ax.scatter(kousi_ax,kousi_ay,kousi_az,s=20,c=ac,)

    bx,by,bz = a_to_b(a1,a2,a3)
    #ax.scatter(kousi_bx,kousi_by,kousi_bz,s=20,c=bc)
    return bx, by, bz

def cube(r,genten): # 球面の作図
    theta = np.linspace(start=0.0, stop=2.0*np.pi, num=151)
    phi = np.linspace(start=0.0, stop=2.0*np.pi, num=151)
    Theta,Phi = np.meshgrid(theta,phi)
    X = r * np.sin(Theta) * np.cos(Phi) + genten[0]
    Y = r * np.sin(Theta) * np.sin(Phi) + genten[1]
    Z = r * np.cos(Theta) + genten[2]
    ax.plot_wireframe(X,Y,Z,alpha=0.3)

# ベクトル描写関数
def visual_vector_3d(genten,vector,color = 'red'):
    ax.quiver(genten[0], genten[1], genten[2],
              vector[0], vector[1], vector[2],
              color = color, length = 1, arrow_length_ratio = 0.05)

N = 6 # 繰り返し回数設定
k_i = np.array([2,0,0]) # 入射波数
r = np.sqrt((k_i[0]*k_i[0])+(k_i[1]*k_i[1])+(k_i[2]*k_i[2])) 

# 実空間の基本ベクトル
a1 = np.array([2,0,2])
a2 = np.array([2,2,0])
a3 = np.array([0,2,2])

# 実格子点の初期化
kousi_ax = [0 for i in range(N*N*N)]
kousi_ay = [0 for i in range(N*N*N)]
kousi_az = [0 for i in range(N*N*N)]

for i in range(N):  # 実格子点作成
    for j in range(N):
        for k in range(N):
            kousi_ax[N*N*i + N*j + k] = a1[0]*k + a2[0]*j + a3[0]*i
            kousi_ay[N*N*i + N*j + k] = a1[1]*k + a2[1]*j + a3[1]*i
            kousi_az[N*N*i + N*j + k] = a1[2]*k + a2[2]*j + a3[2]*i

k_f = ([0,0,0])
for theta_degree in range(0,360,1):
    bx,by,bz = ab_rotate(a1,a2,a3,theta_degree=theta_degree)
    kijun = int(N*N*N/2 + N*N/2 + N/2)
    genten = np.array([bx[kijun]-k_i[0],by[kijun]-k_i[1],bz[kijun]-k_i[2]])
    dr = r/100              # 許容精度
    for i in range(N*N*N):  # すべての逆格子点に対して
        if i != kijun:      # 基準点を除く
            r_bx = bx[i]-genten[0]
            r_by = by[i]-genten[1]
            r_bz = bz[i]-genten[2]
            r_b = np.sqrt(r_bx*r_bx+r_by*r_by+r_bz*r_bz) # 原点からの距離
            if r-dr < r_b < r+dr:
                k_f = ([r_bx,r_by,r_bz])        # 散乱波の波数
                print(theta_degree)             # 実格子の回転角
                h = i%(N)
                k = ((i-h)%(N*N))/(N)
                l = (i-h-k*N)/(N*N)
                print(h,k,l)
                break
    if k_f != ([0,0,0]):        # 一つ見つかれば終了
        break

if k_f != ([0,0,0]):
    ax.scatter(bx,by,bz,s=10,c='green')                         # 逆格子点
    cube(r=r,genten=genten)                                     # 球面描写
    visual_vector_3d(genten=genten,vector=k_i)                  # 入射波数描写
    visual_vector_3d(genten=genten,vector=k_f,color='orange')   # 散乱波波数描写
    G = k_f - k_i
    visual_vector_3d(genten=(bx[kijun],by[kijun],bz[kijun]),vector=G,color='black')
    print(k_i,k_f)
    print(k_f - k_i)
    
    plt.show()
else:
    print('None')       #発見ならず

#plt.show()


