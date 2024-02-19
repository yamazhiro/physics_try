import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collections import namedtuple
import matplotlib.animation as animation

def plot(results):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    for r in results:
        # namedtupleの各フィールドにはドットでアクセス
        ax.plot(r.yt, r.zt, marker='.',markersize=10, label=r.key)
    plt.grid(which='major',lw=0.4)
    ax.set_xlim(-0.1,0.1)
    ax.set_ylim(-0.1,0.1)
    ax.set_xlabel(r'Slide$(y-axis)$ [m]',fontsize=20)
    ax.set_ylabel(r'Hop$(z-axis)$ [m]',fontsize=20)
    ax.set_aspect('equal')
    ax.legend(ncol=2,fontsize=11)

def plot3d(results):
    fig,ax=plt.subplots(figsize=(8,8),subplot_kw={'projection':'3d'})
    #ax.set_title("", size = 20)
    ax.set_xlabel(r'$x$ [m]', size = 14)
    ax.set_ylabel(r'$y$ [m]', size = 14)
    ax.set_zlabel(r'$z$ [m]', size = 14)
    for r in results:
        # namedtupleの各フィールドにはドットでアクセス
        ax.plot(r.xt, r.yt, r.zt, marker='.', label=r.key)
    #ax.set_aspect('equal')
    ax.set_yticks([-0.5,0,0.5])
    ax.set_xlim(0,18)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(0,2)
    fig.legend(ncol=1,fontsize=11)

def anim(results,t):
    fig,ax = plt.subplots(figsize=(8,8),subplot_kw={'projection':'3d'})
    ax.set_xlabel(r'$x$ [m]', size = 14)
    ax.set_ylabel(r'$y$ [m]', size = 14)
    ax.set_zlabel(r'$z$ [m]', size = 14)
    plt.gca().set_aspect('equal')
    ax.set_yticks([-0.5,0,0.5])
    ax.set_xlim(0,18)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(0,2)
    ims = []
    for r in results:
        for j in range(len(t)):
            im = ax.plot(r.xt[j],r.yt[j],r.zt[j], marker='.',c='blue', label=r.key)
            im2 = ax.plot(r.xt[:j],r.yt[:j],r.zt[:j], marker=None, ls='--',lw=1,c='blue', label=r.key)  # 軌跡作成
            text = ax.text(0.05, 1.05,1.05, f"t = {t[j]:.2f}", transform=ax.transAxes)
            ims.append(im+im2+[text])
    anim = animation.ArtistAnimation(fig,ims,interval=50,repeat=False)
    anim.save('kidou02.gif', writer="pillow")

def f_newton(t, X, m, g, b, c, w_x, w_y, w_z):
    # X = [x, v_x, y, v_y, z, v_z]
    dXdt = np.array([
        X[1],                   # dx/dt = v_x
        -(b/m) * X[1] + (c/m)*(w_y*X[5] - w_z*X[3]), # dv_x/dt = -(b/m) v_x + (c/m) (w_y v_z - w_z v_y)
        X[3],                   # dy/dt = v_y
        -(b/m) * X[3] + (c/m)*(w_z*X[1] - w_x*X[5]), # dv_y/dt = -(b/m) v_y + (c/m) (w_z v_x - w_x v_z)
        X[5],                   # dz/dt = v_z
        -(b/m) * X[5] - g + (c/m)*(w_x*X[3] - w_y*X[1])  # dv_z/dt = -(b/m) v_z - g + (c/m) (w_x v_y - w_y v_x)
    ])
    return dXdt

def solve_newton(eq_params, X0, t_range, n_t):
    m, g, b, c, w_x, w_y, w_z = eq_params
    
    sol = solve_ivp(f_newton, t_range, X0, args=(m, g, b, c, w_x, w_y, w_z), dense_output=True)
    print(sol.message)
    
    t_start, t_end = t_range
    t = np.linspace(t_start, t_end, n_t)
    Xt = sol.sol(t)
    assert Xt.shape == (6, n_t)
    
    return Xt[0, :], Xt[1, :], Xt[2, :], Xt[3, :], Xt[4, :], Xt[5, :]

def pitching(theta=0,g=9.8,m=0.1,b=0.1,c=0.0001,rps=150,w=np.array([0,1,0]),v0=100.0/3.6,
             results=[],results2d=[],kijun=[],
             t_start=0,t_end=5.0,n_t=101,Result=[],Result2d=[]):
    if np.linalg.norm(w,ord=2) == 0:
        normalize_w = np.array([0,0,0])
    else:
        normalize_w = w /np.linalg.norm(w,ord=2)
    rotate_x = rps*normalize_w[0]
    rotate_y = rps*normalize_w[1]
    rotate_z = rps*normalize_w[2]
    label = f'rps={round(rps,1)}, axis={[w[0],w[1],w[2]]}'
    w_x = rotate_x*2*np.pi
    w_y = rotate_y*2*np.pi
    w_z = rotate_z*2*np.pi
    # X0 = [x0, v0_x, y0, v0_y, z0, v0_z]
    X0 = np.array([0, v0 * np.cos(theta), 0, 0, 2, v0 * np.sin(theta)])
    
    xt, _, yt, _, zt, _ = solve_newton(eq_params=(m, g, b, c, w_x, w_y, w_z),
                                    X0=X0, t_range=(t_start, t_end), n_t=n_t)
    # 結果をnamedtupleにまとめ、リストに追加 (*2)
    for i in range(len(xt)):
        if xt[i] > 20:
            y = ((20-xt[i-1])*yt[i]+(xt[i]-20)*yt[i-1])/(xt[i]-xt[i-1])
            z = ((20-xt[i-1])*zt[i]+(xt[i]-20)*zt[i-1])/(xt[i]-xt[i-1])
            results2d.append(Result2d(str(label), y-kijun[0], z-kijun[1]))
            break
    results.append(Result(str(label), xt, yt, zt))

# マグヌス効果がないとき，x=16.5 mでのy,z座標の取得
def pitching0(theta=0,g=9.8,m=0.1,b=0.1,c=0.0001,v0=100.0/3.6,
             t_start=0,t_end=5.0,n_t=101):
    # X0 = [x0, v0_x, y0, v0_y, z0, v0_z]
    X0 = np.array([0, v0 * np.cos(theta), 0, 0, 2, v0 * np.sin(theta)])
    
    xt, _, yt, _, zt, _ = solve_newton(eq_params=(m, g, b, c, 0, 0, 0),
                                    X0=X0, t_range=(t_start, t_end), n_t=n_t)
    # 結果をnamedtupleにまとめ、リストに追加 (*2)
    for i in range(len(xt)):
        if xt[i] > 20:
            y = ((20-xt[i-1])*yt[i]+(xt[i]-20)*yt[i-1])/(xt[i]-xt[i-1])
            z = ((20-xt[i-1])*zt[i]+(xt[i]-20)*zt[i-1])/(xt[i]-xt[i-1])
            break
    return y,z

def plot_pitching(theta=0,g=9.8,m=0.1,b=0.1,c=0.0001,
                  rps=150,w=np.array([0,1,0]),v0=100.0/3.6,
                  t_start=0,t_end=5.0,n_t=101,Result=[],Result2d=[]):
    results = []
    results2d = []
    y0,z0 = pitching0(theta=theta,g=g,m=m,b=b,c=c,v0=v0,t_start=t_start,t_end=t_end,n_t=n_t)
    kijun = [y0,z0]
    
    pitching(theta=theta,g=g,m=m,b=b,c=c,rps=rps,w=w,v0=v0,
                results=results,results2d=results2d,kijun=kijun,
                t_start=t_start,t_end=t_end,n_t=n_t,Result=Result,Result2d=Result2d)
        
    #plot(results2d)
    #plot3d(results)   #放物線の描画
    t=[i*t_end/n_t for i in range(0,n_t)]
    anim(results,t)
    

def main():
    # パラメーター (MKS単位系: m, kg, s)
    g = 9.8  # 重力加速度[m/s^2]
    m = 0.15  # 質量[kg]
    b = 0.1  # [kg/s]
    c = 0.00005 # [kg]
    
    rps= 80
    w = np.array([0,-1,0])

    v0 = 150 * 1000 / 3600  # 初速 [m/s]
    
    t_start = 0  # 初期時刻 [s]
    t_end = 0.6  # 最終時刻 [s]
    n_t = 61  # 時刻の刻み数（グラフ描画用）
    # 投射仰角
    theta=0 # 投射仰角[degree]
    
    # 各初期条件に対する結果をそれぞれnamedtupleとしてまとめる (*1)
    Result = namedtuple('Result', ['key', 'xt', 'yt', 'zt'])
    Result2d = namedtuple('Result2d', ['key', 'yt', 'zt'])

    plot_pitching(g=g,m=m,b=b,c=c,rps=rps,w=w,
                  t_start=t_start,t_end=t_end,n_t=n_t,
                  v0=v0,theta=theta,Result=Result,Result2d=Result2d)

if __name__ == '__main__':
    main()
    plt.show()
