import requests, math, io, sys, cv2, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# Own modules
from som import SOM


def process_color_maps(n_colors, img_dir, max_size=420):
    # This is the main processing flow for downloading image and 
    # processing the color palette. 
    for filename in os.listdir(img_dir):
        file = os.path.join(img_dir, filename)
        if os.path.isfile(file):
            print(filename)
            img = get_image(file,max_size)
            if img is None:
                return
            kmeans = k_means(img, n_colors)
            gs = plot_3d_pixels(img, kmeans)
            plot_kmeans_palette(kmeans, n_colors, gs)
            som_plot(img, n_colors, gs)
            #saving in a sub-dir
            plt.savefig(img_dir+'/output/Info_'+filename)
            plt.show()

def get_image(loc,size):
    # Method for getting the img, scaling the resolution and color values.
    max_size = size
    try:
        img = np.asarray(Image.open(loc))
    except:
        print('Failed to parse image')
        print(sys.exc_info()[0])
        return None

    scaling = max_size / max(img.shape[0], img.shape[1])
    img = cv2.resize(img, (0,0), fx=scaling, fy=scaling)
    img = img.astype(np.float32)
    img /= 255.
    return img

def k_means(img, n_colors):
    # Method for reducing the image color palette to n_colors 
    # using K-means clustering algorithm. 

    w, h, d = original_shape = tuple(img.shape)
    img_vect = np.reshape(img, (w * h, d))
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(img_vect)
    labels = kmeans.predict(img_vect)

    def recreate_image(codebook, labels, w, h):
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image

    proc_img = recreate_image(kmeans.cluster_centers_, labels, w, h)

    # Plot the original image and the reduced palette image. 
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[0].set_title('Original image')
    axes[0].axis('off')
    axes[1].imshow(proc_img)
    axes[1].set_title('Quantized image')
    axes[1].axis('off')
    plt.show()

    return kmeans

def plot_3d_pixels(img, kmeans):
    # Method for plotting 3D pixel cloud of the original image pixels. 
    # The K-means cluster center points are plotted on the grap, 

    # Select max 5000 pixels for the plot. 
    n_pixels = img.shape[0]*img.shape[1] if img.shape[0]*img.shape[1] < 5000 else 5000
    flat = shuffle(img.reshape((img.shape[0]*img.shape[1], 3)))[:n_pixels]
    reds = flat[:, 0]
    greens = flat[:, 1] 
    blues = flat[:, 2]
    colors = flat

    print('=' * 80)
    print('K-means cluster centers in RGB space:')

    # Plot the original image pixels.
    fig = plt.figure(figsize=(15,10))
    gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.07)
    fig.tight_layout()
    fig.set_facecolor('#fff')
    # ax = fig.add_subplot(111, projection='3d')
    # ax = plt.subplot(221, projection='3d')
    ax = plt.subplot(gs.new_subplotspec((0,0), colspan=2, rowspan=3), projection='3d')
    ax.scatter(reds, greens, blues, c=colors, alpha=1, s=10, marker='.', zorder=1)

    # Plot the K-means clustering center points. 
    xs = kmeans.cluster_centers_[:, 0]
    ys = kmeans.cluster_centers_[:, 1]
    zs = kmeans.cluster_centers_[:, 2]
    coolors = kmeans.cluster_centers_
    ax.scatter(xs, ys, zs, c=coolors, marker='o', alpha=1, s=500, edgecolors=(0,0,0), zorder=100)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    return gs
    # plt.savefig(curr_file) #at first i wanted each part seperated.

def plot_kmeans_palette(kmeans, n_colors,gs):
    # Method for plotting the palette from K-means clustering algorithm. 
    # The palette is sorted and the color RGB values are printed. 

    color_palette = np.ones(( n_colors*10, 10,3)) #switched to vertical palette
    clusters = kmeans.cluster_centers_

    def colsort(arr, col_index, tol):
        # Method to calculate that is the color mainly red, green or blue (very coarsely).
        if col_index == 0: # Red
            comp1 = 1
            comp2 = 2
        elif col_index == 1: # Green
            comp1 = 0
            comp2 = 2
        else: # Blue
            comp1 = 0
            comp2 = 1
        res = (arr[:, col_index] > arr[:, comp1] + tol) * (arr[:, col_index] > arr[:, comp2] + tol)
        res_pal = arr[res]
        res_pal = res_pal[res_pal[:, col_index].argsort()]  
        return res, res_pal

    # Split the palette to R, G, B and gray colors. 
    tol = 0.1
    reds, red_pal = colsort(clusters, 0, tol) 
    greens, green_pal = colsort(clusters, 1, tol) 
    blues, blue_pal = colsort(clusters, 2, tol) 
    grays = (reds + greens + blues) == False
    gray_pal = clusters[grays]
    gray_pal = gray_pal[gray_pal[:, 2].argsort()]

    green_pal = np.flip(green_pal, axis=0)
    #blue_pal = np.flip(blue_pal, axis=0)
    gray_pal = np.flip(gray_pal, axis=0)

    # Put the palette elements together
    palette = np.concatenate([red_pal, green_pal, blue_pal, gray_pal])
    tickPlacement = []
    for i in range(palette.shape[0]):
        color_palette[i*10:(i+1)*10,:] = palette[i]
        tickPlacement.append(i*10+5)

    plt.subplot(gs.new_subplotspec((0,2), colspan=1, rowspan=1))
    plt.imshow(color_palette)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.yaxis.tick_right()
    ax.set_ylabel('Palette')
    ax.set_anchor('SW')
    # plt.savefig(curr_file)

    # Print the RGB values for the palette. This happens most beautifully
    # by printing the data in Pandas DataFrame. 
    rgb_df = pd.DataFrame(np.zeros((n_colors)))
    hex_df = pd.DataFrame(np.zeros((n_colors)))
    palette *= 255
    palette = palette.astype(np.int)
    colors = []

    for i in range(n_colors):
        rgb_df.iloc[i] = str(palette[i])
        hex_color = '#{:02x}{:02x}{:02x}'.format(palette[i, 0], palette[i, 1], palette[i, 2])
        hex_df.iloc[i] = hex_color
        color = '{:02x}{:02x}{:02x}'.format(palette[i, 0], palette[i, 1], palette[i, 2])
        colors.append(color)
    # print('RGB values:')
    # print(rgb_df.T)
    print('\n')
    print('HEX color values:')
    print(hex_df.T)

    #getting color names from the api and appending them to the palette
    PARAMS = {'values': ','.join(colors), 'goodnamesonly': 'true'}
    r = requests.get(url='https://api.color.pizza/v1/', params=PARAMS)
    data = r.json()
    colorNames = []

    for color in data['colors']:
        colorNames.append(color['name'])
    ax.set_yticks(tickPlacement, colorNames)
    print(' ,'.join(colorNames))

def som_plot(img, som_dim,gs):
    # Method for creating a small Kohonen Self-Organizing Map (SOM) of the image and
    # plotting the SOM on the screen. RGB values of the colors are printed on the screen. 

    # Select small amount of random pixels from the image. 
    n_pixels = 500
    colors = shuffle(img.reshape((img.shape[0]*img.shape[1], 3)))[:n_pixels]
    
    print('\n')
    print('=' * 80)
    print('Self-Organized Map of {} randomly picked:'.format(som_dim*som_dim))
    
    # Train the SOM model with small amount of iterations. 
    som = SOM(som_dim, som_dim, 3, 10)
    som.train(colors)

    #Get output grid from the SOM. This is plotted as color palette. 
    image_grid = som.get_centroids()
    plt.subplot(gs.new_subplotspec((1,2), colspan=1, rowspan=2))
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.imshow(image_grid)
    plt.title('Color map', y=-0.1)
    # plt.savefig(curr_file)
    # plt.show()

    # Create RGB palette values from the image_grid.
    grid = np.array(image_grid)
    grid *= 255
    grid = grid.astype(np.int)
    rgb_df = pd.DataFrame(np.zeros((som_dim, som_dim)))
    hex_df = pd.DataFrame(np.zeros((som_dim, som_dim)))

    for i in range(som_dim):
        for j in range(som_dim):
            rgb_df.iloc[i,j] = str(grid[i, j])
            hex_color = '#{:02x}{:02x}{:02x}'.format(grid[i, j, 0], grid[i, j, 1], grid[i, j, 2])
            hex_df.iloc[i,j] = hex_color

    # print('RGB values:')
    # print(rgb_df)
    print('\n')
    print('HEX color values:')
    print(hex_df)
