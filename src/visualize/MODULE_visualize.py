import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

def plot_curve_and_points(points, plot_title):
    def scaled_expo(x,C,lamb):
        return C*np.exp(-lamb*x)
    x_vals = np.array([0.3, 0.9, 1.5, 2.1, 2.7])
    params, _ = curve_fit(scaled_expo, x_vals, points)
    ppoints = [(x,y) for x,y in zip(x_vals, points)]
    print(f'Fitting a curve on {ppoints}')
    print(f"C = {params[0]}, \u03BB = {params[1]}")
    fig, axs = plt.subplots(1, 1)
    plt.ylim(0,1)
    axs.scatter(x_vals, points)
    x_curve = np.linspace(0, x_vals.max(), 200)
    axs.plot(x_curve, scaled_expo(x_curve, *params), color = 'red', alpha = 0.8)
    axs.text(1.2, 0.9, '$f(x) = C \\cdot e^{- \lambda x} \\to %.2f e^{-%.2fx}$' % (params[0], params[1]), fontsize=14)
    plt.title(plot_title)
    plt.show()

def compare_curves_on_the_same_image(points1, model_name1, points2, model_name2, image_name):
    def scaled_expo(x,C,lamb):
        return C*np.exp(-lamb*x)
    x_vals = np.array([0.3, 0.9, 1.5, 2.1, 2.7])
    x_curve = np.linspace(0, x_vals.max(), 200)

    params1, _ = curve_fit(scaled_expo, x_vals, points1)
    params2, _ = curve_fit(scaled_expo, x_vals, points2)

    print(f'Fitting a curve on {model_name1}\'s prediction probabilities: {points1}')
    print(f'Fitting a curve on {model_name2}\'s prediction probabilities: {points2}')

    fig, axs = plt.subplots(1, 2, figsize = (12, 5))
    
    axs[0].set_ylim(0, 1)
    axs[0].scatter(x_vals, points1)
    axs[0].plot(x_curve, scaled_expo(x_curve, *params1), color = 'red', alpha = 0.8)
    C1 = round(params1[0], 4)
    lamb1 = round(params1[1], 4)
    axs[0].set_title(f'{model_name1}: C = {C1}, $\lambda$ = {lamb1}')
    axs[0].text(1.2, 0.9, '$f(x) = %.2f e^{-%.2fx}$' % (params1[0], params1[1]), fontsize=14)

    axs[1].sharey(axs[0])
    axs[1].scatter(x_vals, points2)
    axs[1].plot(x_curve, scaled_expo(x_curve, *params2), color = 'red', alpha = 0.8)
    C2 = round(params2[0], 4)
    lamb2 = round(params2[1], 4)
    axs[1].set_title(f'{model_name2}: C = {C2}, $\lambda$ = {lamb2}')
    axs[1].text(1.2, 0.9, '$f(x) = %.2f e^{-%.2fx}$' % (params2[0], params2[1]), fontsize=14)

    fig.suptitle(f'Fitted curve comparison on the same image: {image_name}')
    plt.show()



def plot_single_prediction_and_lambda(csv_path, model_name, class_name, show = False, save_path = None):    
    image_name = f'{model_name}_on_{class_name}'
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(2,2, figsize = (15, 10))

    xy = np.vstack([df['correct_idx'], df['lambda']])
    z = gaussian_kde(xy)(xy)
    #z = gaussian_kde(df['lambda'])(df['lambda'])

    idxs = z.argsort()
    x = df['correct_idx'][idxs]
    y = df['lambda'][idxs]
    z = z[idxs]
    
    AA = ax[0, 0].scatter(x, y, s=100, c = z, cmap = plt.cm.get_cmap('rainbow', 20))
    ax[0, 0].set_xlabel('correct_idx')
    ax[0, 0].set_ylabel('$\lambda$')
    
    cb = fig.colorbar(AA, label = 'density of $\lambda$', ax = ax[0, 0])
    cb.set_ticks([])
    
    
    to_plt_lambda = df[df['correct_idx'] == -1]['lambda']
    ax[0, 1].hist(to_plt_lambda, 100)
    ax[0, 1].set_title('Distribution of $\lambda$ for wrong predictions')
    ax[0, 1].set_xlabel('$\lambda$')
    ax[0, 1].set_ylabel('# images')
    

    to_plt_lambda = df[df['correct_idx'] == 0]['lambda']
    ax[1, 0].hist(to_plt_lambda, 100)
    ax[1, 0].set_title('Distribution of $\lambda$ for rank 1 predictions')
    ax[1, 0].sharex(ax[0, 1])
    
    ax[1, 0].set_xlabel('$\lambda$')
    ax[1, 0].set_ylabel('# images')
    
    lst1 = []
    for i in range(1, 5):
        lst1 += list(df[df['correct_idx'] == i]['lambda'])
    ax[1, 1].hist(lst1, 100)
    ax[1, 1].set_title('Distribution of $\lambda$ for non rank 1 predictions')
    ax[1, 1].sharex(ax[1, 0])
    ax[1, 1].set_xlabel('$\lambda$')
    ax[1, 1].set_ylabel('# images')
    
    fig.suptitle(image_name, fontsize=16)
    
    if show:
        plt.show()
    if save_path != None:
        fig.savefig(f'{save_path}/{image_name}.png')
