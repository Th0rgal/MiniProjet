# Load general libraries
import glob
import inspect
import os
import math
from typing import Callable, List, Optional, Iterable, Union
import warnings
import itertools
from os import walk

from pygments import highlight  # type: ignore
from pygments.lexers import PythonLexer  # type: ignore
from pygments.formatters import HtmlFormatter  # type: ignore
from IPython.display import display, HTML  # type: ignore

from PIL import Image  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore
from sklearn.neighbors import KNeighborsClassifier

from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # type: ignore
from matplotlib.figure import Figure  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.colors import LinearSegmentedColormap  # type: ignore

import librosa  # for working with audio in python
import librosa.display  # for waveplots, spectograms, etc
import soundfile as sf  # for accessing file information
import IPython.display as ipd  # for playing files within python

import seaborn as sns  # type: ignore

sns.set()
warnings.simplefilter(action="ignore", category=FutureWarning)


black_red_cmap = LinearSegmentedColormap.from_list("black_red_cmap", ["black", "red"])
black_green_cmap = LinearSegmentedColormap.from_list(
    "black_green_cmap", ["black", "green"]
)
black_blue_cmap = LinearSegmentedColormap.from_list(
    "black_blue_cmap", ["black", "blue"]
)


def show_color_channels(img: Image.Image) -> Figure:
    """
    Return a figure displaying the image together with its red, green, and blue layers
    """
    M = np.array(img)
    fig = Figure(figsize=(30, 5))
    (ax, axr, axg, axb) = fig.subplots(1, 4)  # Quatre zones de dessin
    # Dessin de l'image et de ses trois couches
    ax.imshow(M)
    imgr = axr.imshow(M[:, :, 0], cmap=black_red_cmap, vmin=0, vmax=255)
    imgg = axg.imshow(M[:, :, 1], cmap=black_green_cmap, vmin=0, vmax=255)
    imgb = axb.imshow(M[:, :, 2], cmap=black_blue_cmap, vmin=0, vmax=255)
    # Ajout des barres d'échelle de couleur aux images
    fig.colorbar(imgr, ax=axr)
    fig.colorbar(imgg, ax=axg)
    fig.colorbar(imgb, ax=axb)

    return fig


def foreground_filter(
    img: Union[Image.Image, np.ndarray], theta: int = 150
) -> np.ndarray:
    """Create a black and white image outlining the foreground."""
    M = img if type(img) == np.ndarray else np.array(img)
    G = np.min(M[:, :, 0:3], axis=2)
    return G < theta


def transparent_background_filter(
    img: Union[Image.Image, np.ndarray], theta: int = 150
) -> Image.Image:
    """Create a cropped image with transparent background."""
    F = foreground_filter(img, theta=theta)
    M = np.array(img)
    N = np.zeros([M.shape[0], M.shape[1], 4], dtype=M.dtype)
    N[:, :, :3] = M[:, :, :3]
    N[:, :, 3] = F * 255
    return Image.fromarray(N)


def darkness(img: Image.Image) -> float:
    """Return the darkness of an image"""
    M = np.array(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    B = M[:, :, 2] * 1.0
    return np.mean(R + G + B)


def redness(img: Image.Image) -> float:
    """Return the redness of a PIL image."""
    M = np.array(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    return np.mean((R - G)[foreground_filter(img)])


def elongation(img: Image.Image) -> float:
    """Extract the scalar value elongation from a PIL image."""
    F = foreground_filter(img)
    # Build the cloud of points given by the foreground image pixels
    xy = np.argwhere(F)
    # Center the data
    C = np.mean(xy, axis=0)
    Cxy = xy - np.tile(C, [xy.shape[0], 1])
    # Apply singular value decomposition
    U, s, V = np.linalg.svd(Cxy)
    return s[0] / s[1]


def elongation_plot(img: Image.Image, subplot: Axes) -> None:
    """Plot the principal axes of the SVD when computing the elongation"""
    # Build the cloud of points defined by the foreground image pixels
    F = foreground_filter(img)
    xy = np.argwhere(F)
    # Center the data
    C = np.mean(xy, axis=0)
    Cxy = xy - np.tile(C, [xy.shape[0], 1])
    # Apply singular value decomposition
    U, s, V = np.linalg.svd(Cxy)

    N = len(xy)
    a0 = s[0] / np.sqrt(N)
    a1 = s[1] / np.sqrt(N)

    # Plot the center
    subplot.plot(
        C[1], C[0], "ro", linewidth=50, markersize=10
    )  # x and y are j and i in matrix coord.
    # Plot the principal axes
    subplot.plot(
        [C[1], C[1] + a0 * V[0, 1]], [C[0], C[0] + a0 * V[0, 0]], "r-", linewidth=3
    )
    subplot.plot(
        [C[1], C[1] - a0 * V[0, 1]], [C[0], C[0] - a0 * V[0, 0]], "r-", linewidth=3
    )
    subplot.plot(
        [C[1], C[1] + a1 * V[1, 1]], [C[0], C[0] + a1 * V[1, 0]], "g-", linewidth=3
    )
    subplot.plot(
        [C[1], C[1] - a1 * V[1, 1]], [C[0], C[0] - a1 * V[1, 0]], "g-", linewidth=3
    )


def load_images(datadir: str, pattern: str = "*.png") -> pd.Series:
    """
    Return all the images in `datadir` whose name match the pattern

    The images are returned as a Panda Series, with the image file
    names as indices.

    Example:

    This returns all png images in `dataset/` whose name starts
    with `a`:

        >>> images = load_images('dataset', 'a*.png')

    The names of the files serve as index:

        >>> images.index
    """
    paths = sorted(glob.glob(os.path.join(datadir, pattern)))
    images = [Image.open(path) for path in paths]
    names = [os.path.basename(path) for path in paths]
    return pd.Series(images, names)


def image_grid(
    images: List[Image.Image], columns: int = 5, titles: Optional[Iterable] = None
) -> Figure:
    """
    Return a figure holding the images arranged in a grid

    Optionally the number of columns and/or image titles can be provided.

    Example:

         >>> image_grid(images)
         >>> image_grid(images, titles=[....])

    """
    rows = math.ceil(1.0 * len(images) / columns)
    fig = Figure(figsize=(10, 10.0 * rows / columns))
    if titles is None:
        titles = range(len(images))
    for k, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(rows, columns, k + 1)
        ax.imshow(img)
        ax.tick_params(axis="both", labelsize=0, length=0)
        ax.grid(b=False)
        ax.set_xlabel(title, labelpad=-4)
    return fig


def color_histogram(img: Image.Image) -> Figure:
    """
    Return a histogram of the color channels of the image
    """
    M = np.array(img)
    n, p, m = M.shape
    MM = np.reshape(M, (n * p, m))
    if m == 4:  # Discard transparency channel if present
        MM = MM[:, 0:3]
    colors = ["red", "green", "blue"]
    fig = Figure(figsize=(12, 6))
    ax = fig.add_subplot()
    ax.hist(MM, bins=10, density=True, histtype="bar", color=colors, label=colors)
    ax.set_xlabel("Pixel amplitude in each color channel")
    ax.set_ylabel("Pixel density")
    return fig


def transparent_background(img: Image.Image) -> Image.Image:
    """Sets the white background of an image to transparent"""
    data = img.getdata()  # Get a list of tuples
    newData = []
    for a in data:
        a = a[:3]  # Shorten to RGB
        if np.mean(np.array(a)) == 255:  # the background is white
            a = a + (0,)  # Put a transparent value in A channel (the fourth one)
        else:
            a = a + (255,)  # Put a non- transparent value in A channel
        newData.append(a)
    img.putdata(newData)  # Get new img ready
    return img


def ls(dir: str) -> List[str]:
    return sorted(os.path.basename(path) for path in glob.glob(os.path.join(dir, "*")))


def split_data(X, Y, verbose=True, seed=0):
    """Make a 50/50 training/test data split (stratified).
    Return the indices of the split train_idx and test_idx."""
    SSS = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    ((train_index, test_index),) = SSS.split(X, Y)
    if verbose:
        print("TRAIN:", train_index, "TEST:", test_index)
    return (train_index, test_index)


def make_scatter_plot(
    df,
    images,
    train_index=[],
    test_index=[],
    filter=None,
    predicted_labels=[],
    show_diag=False,
    axis="normal",
    feat=None,
    theta=None,
) -> Figure:
    """This scatter plot function allows us to show the images.

    predicted_labels can either be:
                    - None (queries shown as question marks)
                    - a vector of +-1 predicted values
                    - the string "GroundTruth" (to display the test images).
    Other optional arguments:
            show_diag: add diagonal dashed line if True.
            feat and theta: add horizontal or vertical line at position theta
            axis: make axes identical if 'square'."""
    fruit = np.array(["B", "A"])

    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot()

    nsample, nfeat = df.shape
    if len(train_index) == 0:
        train_index = range(nsample)
    # Plot training examples
    x = df.iloc[train_index, 0]
    y = df.iloc[train_index, 1]
    f = images.iloc[train_index]
    ax.scatter(x, y, s=750, marker="o", c="w")

    for x0, y0, img in zip(x, y, f):
        ab = AnnotationBbox(OffsetImage(img), (x0, y0), frameon=False)
        ax.add_artist(ab)

    # Plot test examples
    x = df.iloc[test_index, 0]
    y = df.iloc[test_index, 1]

    if len(predicted_labels) > 0 and not (predicted_labels == "GroundTruth"):
        label = (predicted_labels + 1) / 2
        ax.scatter(x, y, s=250, marker="s", color="c")
        for x0, y0, lbl in zip(x, y, label):
            ax.text(
                x0 - 0.03,
                y0 - 0.03,
                fruit[int(lbl)],
                color="w",
                fontsize=12,
                weight="bold",
            )
    elif predicted_labels == "GroundTruth":
        f = images.iloc[test_index]
        ax.scatter(x, y, s=500, marker="s", color="c")
        for x0, y0, img in zip(x, y, f):
            ab = AnnotationBbox(OffsetImage(img), (x0, y0), frameon=False)
            ax.add_artist(ab)
    else:  # Plot UNLABELED test examples
        f = images[test_index]
        ax.scatter(x, y, s=250, marker="s", c="c")
        ax.scatter(x, y, s=100, marker="$?$", c="w")

    if axis == "square":
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(f"$x_1$ = {df.columns[0]}")
    ax.set_ylabel(f"$x_2$ = {df.columns[1]}")

    # Add line on the diagonal
    if show_diag:
        ax.plot([-3, 3], [-3, 3], "k--")

    # Add separating line along one of the axes
    if theta is not None:
        if feat == 0:  # vertical line
            ax.plot([theta, theta], [-3, 3], "k--")
        else:  # horizontal line
            ax.plot([-3, 3], [theta, theta], "k--")

    return fig


def make_scatter_plot2(
    df, train_index=[], test_index=[], show_diag=False, axis="normal",
) -> Figure:
    """This scatter plot function allows us to show the images.

    Other optional arguments:
            show_diag: add diagonal dashed line if True.
            feat and theta: add horizontal or vertical line at position theta
            axis: make axes identical if 'square'."""
    fruit = np.array(["B", "A"])

    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot()

    happy = df[df["classe"] == 1]
    sad = df[df["classe"] == -1]

    # Plot happy songs
    nsample, nfeat = happy.shape
    x = happy.iloc[range(nsample), 0]
    y = happy.iloc[range(nsample), 1]
    ax.scatter(x, y, s=750, marker="o", c="r")

    # Plot sad songs
    nsample, nfeat = sad.shape
    x = sad.iloc[range(nsample), 0]
    y = sad.iloc[range(nsample), 1]
    ax.scatter(x, y, s=750, marker="o", c="b")

    if axis == "square":
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(f"$x_1$ = {df.columns[0]}")
    ax.set_ylabel(f"$x_2$ = {df.columns[1]}")

    # Add line on the diagonal
    if show_diag:
        ax.plot([-3, 3], [-3, 3], "k--")

    return fig


def make_scatter_plot3(df):
    happy = df[df["classe"] == 1]
    sad = df[df["classe"] == -1]
    nsample, nfeat = happy.shape

    perm = list(itertools.combinations(range(nfeat - 1), 2))

    row = len(perm) // 5
    fig = Figure(figsize=(30, 6 * (row + 1)))

    for idx, (a, b) in enumerate(perm):

        ax = fig.add_subplot(row + 1, 5, idx + 1)

        x = happy.iloc[range(nsample), a]
        y = happy.iloc[range(nsample), b]
        ax.scatter(x, y, marker="o", c="r")

        x = sad.iloc[range(nsample), a]
        y = sad.iloc[range(nsample), b]
        ax.scatter(x, y, marker="o", c="b")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel(f"$x_1$ = {df.columns[a]}")
        ax.set_ylabel(f"$x_2$ = {df.columns[b]}")
    return fig


def show_source(function: Callable) -> None:
    code = inspect.getsource(function)
    lexer = PythonLexer()
    formatter = HtmlFormatter(cssclass="pygments")
    html_code = highlight(code, lexer, formatter)
    css = formatter.get_style_defs(".pygments")
    html = f"<style>{css}</style>{html_code}"
    display(HTML(html))


def error_rate(solution, prediction):
    return np.sum(solution != prediction) / len(solution)


def plt_compare(musics, f, ax=None):
    if not ax:
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
    for idx, music in enumerate(musics):
        name = musics.index[idx]
        if name[0] == "a":
            color = "red"
        else:
            color = "blue"
        x = np.linspace(0, 3, 2)
        ax.plot(x, [f(music)] * 2, color=color, linewidth=0.7)
    if not ax:
        return fig


def plt_compare2(musics, f):
    fig = Figure(figsize=(30, 24))
    for idx, music in enumerate(musics):
        name = musics.index[idx]
        if name[0] == "a":
            color = "red"
        else:
            color = "blue"
        ax = fig.add_subplot(4, 5, idx + 1)
        y = f(music)
        x = np.linspace(0, 40, len(y))
        ax.plot(x, y, color=color, linewidth=0.5)
    return fig


def spect_cent(m):
    y, sr = m
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent = cent.reshape(len(cent[0]))
    return cent


def mean_amplitude(m) -> float:
    y, sr = m
    return y.mean()


### it sucks
def get_tempo(m) -> float:
    y, sr = m
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return tempo


def spectral_centroid_mean(m) -> float:
    cent = spect_cent(m)
    return cent.mean()


def amplitude_std(m) -> float:
    y, sr = m
    return np.std(y)


def spectral_centroid_std(m):
    cent = spect_cent(m)
    return np.std(cent)


def chroma_stft(m):
    y, sr = m
    y = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    return np.median(tonnetz)


def load_musics(datadir: str, pattern: str = "*.wav") -> pd.Series:
    paths = sorted(glob.glob(os.path.join(datadir, pattern)))
    musics = [librosa.load(path, offset=15, duration=45) for path in paths]
    names = [os.path.basename(path) for path in paths]
    return pd.Series(musics, names)


def load_music_from_ytb(link):
    mypath = "./"
    os.system("./youtube-dl --audio-format wav --extract-audio " + link)
    vidID = link.split("=")[1]
    f = []
    for dirpath, dirnames, filenames in walk(mypath):
        f.extend(filenames)
    for i in range(0, len(f)):
        if ".wav" in f[i] and vidID in f[i]:
            os.rename(f[i], "test_music.wav")
            break
    name = f[i]
    name = name[:name.find(vidID) - 1]
    music = librosa.load("test_music.wav")
    return (pd.Series([music], [name]))


def guessHappiness(m, df, neigh):
    dfT = pd.DataFrame({
        'spectral_centroid' : m.apply(spectral_centroid_mean),
        'tempo' : m.apply(get_tempo),
        'name' : m.index.map(lambda name: name if name[0] == 'a' else name),
    })

    dfstd = dfT
    dfstd = (dfT - df.mean()) / df.std()

    X = dfstd[['spectral_centroid', 'tempo']]
    NAME = dfT['name']

    Ytest_predicted = neigh.predict(X)

    print(f"La musique {NAME[0]} est: ", end="")
    if (Ytest_predicted[0] == 1):
        print ("Heureuse")
    else:
        print ("Triste")


class oneR():
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.is_trained = False  
        self.ig = 0     # Index of the good feature G
        self.w = 1      # Feature polarity
        self.theta = 0  # Threshold on the good feature

    def fit(self, X, Y):
        '''
        This function should train the model parameters.
        
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            Y: Training label matrix of dim num_train_samples * 1.
        Both inputs are panda dataframes.
        '''
        # Compute correlations
        matrix = X.copy()
        matrix["output"] = Y.copy()
        correlations = matrix.corr()["output"]
        del correlations["output"]
        # Select the most correlated feature in absolute value using the last line,
        # and store it in self.ig
        for i in range(1, len(correlations)): # support n >= 1 checks
            if abs(correlations.iloc[i]) > abs(correlations.iloc[self.ig]):
                self.ig = i


        # Get feature polarity and store it in self.w
        # YOUR CODE HERE
        self.w = np.sign(correlations.iloc[self.ig])
        # Fetch the feature values and multiply by polarity
        G = X.iloc[:, self.ig] * self.w
        # Compute the threshold as a mid-point between cluster centers
        self.theta = G.median()
 
        self.is_trained=True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        
        Args:
            X: Test data matrix of dim num_test_samples * num_feat.
        Return:
            Y: Predicted label matrix of dim num_test_samples * 1.
        '''
        # Fetch the feature of interest and multiply by polarity
        G = X.iloc[:,self.ig] * self.w
        # Make decisions according to threshold
        Y = G.copy()
        Y[G < self.theta] = -1
        Y[G >= self.theta] = 1
              
        return Y

class NNClassifier():
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.is_trained = False
        self.space = np.empty(0)

    def fit(self, X, Y):
        '''
        This function should train the model parameters.
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            Y: Training label matrix of dim num_train_samples * 1.
        Both inputs are panda dataframes.
        '''
        self.space = np.empty((len(X), len(X.iloc[0])))
        self.labels = Y
        for j, feature_label in enumerate(X):
            for i, value in enumerate(X[feature_label]):
                self.space[i, j] = value
        self.is_trained = True

    def distance(self, x, y):
        return np.sqrt( np.sum((x-y)**2) )

class FNNClassifier(NNClassifier):

    def predict(self, X):
        matrix = X.T
        output = np.empty(len(X))
        for i, value_name in enumerate(matrix):
            value = np.array(matrix[value_name])
            best_distance = None
            for j, space_value in enumerate(self.space):
                distance = self.distance(value, space_value)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    output[i] = self.labels[j]
        return output

class KNNClassifier(NNClassifier):

    def __init__(self, k=3):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.k = k
        super().__init__()

    def id_max(self, array):
        id_max = 0
        for i in range(len(array)):
            if array[i] == -1:
                return i
            if array[id_max] < array[i]:
                id_max = i
        return id_max

    def predict(self, X):
        matrix = X.T
        pre_output = np.empty((len(X), self.k))
        for i, value_name in enumerate(matrix):
            value = np.array(matrix[value_name])
            best_distances = np.ones(self.k)*(-1)
            for j, space_value in enumerate(self.space):
                id_max = self.id_max(best_distances)
                distance = self.distance(value, space_value)
                if best_distances[id_max] == -1 or distance < best_distances[id_max]:
                    best_distances[id_max] = distance
                    pre_output[i, id_max] = self.labels[j]
        
        output = np.empty(len(X))
        for i, results in enumerate(pre_output):
            # thx stackoverflow: https://stackoverflow.com/a/28736715/10144963
            values, counts = np.unique(results, return_counts=True)
            ind = np.argmax(counts)
            output[i] = values[ind]  # prints the most frequent element

        return output