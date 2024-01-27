# This is a sample Python script.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def print_hi(name):
    print(f'Calculated value is {name}')


def sin_calc(angle):
    pass
    return np.sin(angle * np.pi / 180)


if __name__ == '__main__':

    theta = np.linspace(90, 270, 46, dtype=int)
    phi = np.linspace(90, 270, 46, dtype=int)

    Nx_elements = 5
    Ny_elements = 5
    Dx_spacing = 0.5
    Dy_spacing = 0.5
    L_lambda = 1
    K_wave = 2 * np.pi / L_lambda
    F_element = 1
    Qx_steering = 0
    Qy_steering = 0

    Ax_tapering = np.ones(Nx_elements, dtype=int)
    Ay_tapering = np.ones(Ny_elements, dtype=int)

    Ax_tapering = [0, 1, 1, 0, 0]
    Ay_tapering = [0, 0, 1, 0, 0]

    A_tapering = np.dot(np.reshape(Ax_tapering, (-1, 1)), [Ay_tapering])

    x_vec = np.linspace(0, Nx_elements - 1, Nx_elements, dtype=int)
    y_vec = np.linspace(0, Ny_elements - 1, Ny_elements, dtype=int)
    m_vec = np.linspace(0, np.size(theta) - 1, np.size(theta), dtype=int)
    n_vec = np.linspace(0, np.size(phi) - 1, np.size(phi), dtype=int)

    F_all = np.zeros((np.size(theta), np.size(phi)), dtype='complex_')

    for m in m_vec:
        for n in n_vec:
            F = np.zeros((Nx_elements, Ny_elements), dtype='complex_')
            for j in x_vec:
                for i in y_vec:
                    F[i][j] = A_tapering[i][j] * np.exp(
                        1j * (j - 1) * K_wave * (Dy_spacing * np.sin(theta[m] * np.pi / 180) * np.sin(
                            phi[n] * np.pi / 180) + Qx_steering *np.pi / 180))
                    F[i][j] = F[i][j] * np.exp(
                        1j * (i - 1) * K_wave * (Dy_spacing * np.sin(theta[m] * np.pi / 180) * np.cos(
                            phi[n] * np.pi / 180) + Qy_steering *np.pi / 180))
            F_all[m][n] = np.sum(F)
    F_norm = F_all / np.max(F_all)

    x_ax, y_ax = np.meshgrid(theta, phi, sparse=True)

    # print(F_all)
    # print(A_tapering)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(np.abs(F_norm))

    #ax2 = fig.add_subplot(132, projection='3d')
    #ax2.plot_wireframe(x_ax, y_ax, np.abs(F_norm))

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x_ax, y_ax, np.abs(F_norm), cmap=plt.get_cmap('jet'), antialiased=True)

    plt.show()

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x, values, label='linear')
    # ax.set_title('Simple plot')
    # ax.grid()
    # fig = plt.show()
