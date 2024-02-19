import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collections import namedtuple

def plot(results):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    for r in results:
        # namedtupleの各フィールドにはドットでアクセス
        ax.plot(r.yt, r.zt, marker='.',markersize=10, label=r.key)
    plt.grid(which='major',lw=0.4)
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.set_xlabel(r'Slide$(y-axis)$ [m]',fontsize=20)
    ax.set_ylabel(r'Hop$(z-axis)$ [m]',fontsize=20)
    ax.set_aspect('equal')
    #ax.legend(ncol=2)

def plot3d(results):
    fig,ax=plt.subplots(figsize=(8,4),subplot_kw={'projection':'3d'})
    ax.set_xlabel(r'$x$ [m]', size = 14)
    ax.set_ylabel(r'$y$ [m]', size = 14)
    ax.set_zlabel(r'$z$ [m]', size = 14)
    for r in results:
        # namedtupleの各フィールドにはドットでアクセス
        ax.plot(r.xt, r.yt, r.zt, marker='.', label=r.key)
    ax.set_yticks([-1,0,1])
    ax.set_xlim(0,18)
    ax.set_ylim(-1,1)
    ax.set_zlim(0,2)

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

def x_rot(w=np.array([0,-1,0]),theta_degree=10):
    theta = theta_degree *np.pi/180
    w_x = w[0]
    w_y = w[1]*np.cos(theta) - w[2]*np.sin(theta)
    w_z = w[1]*np.sin(theta) + w[2]*np.cos(theta)
    w_rot=np.array([w_x,w_y,w_z])
    return w_rot

def y_rot(w=np.array([0,-1,0]),theta_degree=10):
    theta = theta_degree *np.pi/180
    w_x = w[0]*np.cos(theta) + w[2]*np.sin(theta)
    w_y = w[1]
    w_z = -w[0]*np.sin(theta) + w[2]*np.cos(theta)
    w_rot=np.array([w_x,w_y,w_z])
    return w_rot

def z_rot(w=np.array([0,-1,0]),theta_degree=10):
    theta = theta_degree *np.pi/180
    w_x = w[0]*np.cos(theta) - w[1]*np.sin(theta)
    w_y = w[0]*np.sin(theta) + w[1]*np.cos(theta)
    w_z = w[2]
    w_rot=np.array([w_x,w_y,w_z])
    return w_rot

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

def plot_pitching(theta=0,g=9.8,m=0.1,b=0.1,c=0.0001,N=1,theta_degree=10,
                  rps=150,w=np.array([0,1,0]),v0=100.0/3.6,
                  t_start=0,t_end=5.0,n_t=101,Result=[],Result2d=[],Kijun=[]):
    results = []
    results2d = []
    y0,z0 = pitching0(theta=theta,g=g,m=m,b=b,c=c,v0=v0,t_start=t_start,t_end=t_end,n_t=n_t)
    kijun = [y0,z0]
    
    for i in range(N):
        w_rot = x_rot(w=w,theta_degree=i*theta_degree)
        pitching(theta=theta,g=g,m=m,b=b,c=c,rps=rps,w=w_rot,v0=v0,
                results=results,results2d=results2d,kijun=kijun,
                t_start=t_start,t_end=t_end,n_t=n_t,Result=Result,Result2d=Result2d)
    
    for i in range(N):
        w_rot = x_rot(w=w,theta_degree=i*theta_degree)
        pitching(theta=theta,g=g,m=m,b=b,c=c,rps=rps-10,w=w_rot,v0=v0,
                results=results,results2d=results2d,kijun=kijun,
                t_start=t_start,t_end=t_end,n_t=n_t,Result=Result,Result2d=Result2d)
    
    for i in range(N):
        w_rot = x_rot(w=w,theta_degree=i*theta_degree)
        pitching(theta=theta,g=g,m=m,b=b,c=c,rps=rps-20,w=w_rot,v0=v0,
                results=results,results2d=results2d,kijun=kijun,
                t_start=t_start,t_end=t_end,n_t=n_t,Result=Result,Result2d=Result2d)
    
    for i in range(N):
        w_rot = x_rot(w=w,theta_degree=i*theta_degree)
        pitching(theta=theta,g=g,m=m,b=b,c=c,rps=rps-30,w=w_rot,v0=v0,
                results=results,results2d=results2d,kijun=kijun,
                t_start=t_start,t_end=t_end,n_t=n_t,Result=Result,Result2d=Result2d)
    
    plot(results2d)
    plot3d(results)   #放物線の描画

def main():
    # パラメーター (MKS単位系: m, kg, s)
    g = 9.8  # 重力加速度[m/s^2]
    m = 0.15  # 質量[kg]
    b = 0.1  # [kg/s]
    c = 0.00005 # [kg]
    
    rps= 40
    w = np.array([0,-1,0])
    N = 8  # 軸の回転回数
    theta_degree = 45 # 軸の回転角度

    v0 = 150 * 1000 / 3600  # 初速 [m/s]
    
    t_start = 0  # 初期時刻 [s]
    t_end = 0.6  # 最終時刻 [s]
    n_t = 31  # 時刻の刻み数（グラフ描画用）
    # 投射仰角
    theta=0 # degree
    
    # 各初期条件に対する結果をそれぞれnamedtupleとしてまとめる (*1)
    Result = namedtuple('Result', ['key', 'xt', 'yt', 'zt'])
    Result2d = namedtuple('Result2d', ['key', 'yt', 'zt'])

    plot_pitching(g=g,m=m,b=b,c=c,rps=rps,w=w,N=N,theta_degree=theta_degree,
                  t_start=t_start,t_end=t_end,n_t=n_t,
                  v0=v0,theta=theta,Result=Result,Result2d=Result2d)
    
if __name__ == '__main__':
    main()
    plt.show()
