#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Stanislav Krč

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any


def integrate(f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000) -> float:
    """
    Aproximates integral of f, on interval <a,b>  with inaccurate rectangle method 

    Args:
        f (Callable[[NDArray], NDArray]): aproximated function
        a (float): left border
        b (float): right border
        steps (int, optional): Number of steps(rectangles) for volume aproximation. Defaults to 1000.

    Returns:
        float: Area between f and x axis
    """
    X = b-a
    oneStep = X/steps
    xPoints = np.linspace(a, b-oneStep, num=steps,
                          endpoint=True, dtype=None, axis=0)
    yPoints = f(((xPoints+oneStep)+xPoints)/2)
    squares = ((oneStep)*yPoints)

    area = np.sum(squares)
    return area


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    """
    Generates,prints and saves (if desired) all lines defined by formula f(x)=a^2*x^3*sinx
    with specific formula and highlited area between f and y axis 

    Args:
        a (List[float]): List containing set of numbers
        show_figure (bool, optional): If true, shows plot in new window. Defaults to False.
        save_path (str | None, optional): If set, creates (or updates) file with plot image. Defaults to None.
    """
    x = np.arange(-3, 3.01, 0.01)

    axis = 1

    fa = (np.float_power(np.reshape(a, (len(a), 1)).swapaxes(axis, -1), 2)
          * np.float_power(x, 3)*np.sin(x)).swapaxes(axis, -1)

    plt.close()
    lines = plt.plot(x, fa.T, linewidth=2)

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
    plt.rc

    labels = list(map(lambda n: r"$y_{{{}}}(x)$".format(n), a))

    plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12),
               fancybox=True, ncols=(len(a)))

    labelText = r"$\int{f_{s1}(x)}dx=$"
    # go through all elements in a, generate area under fa[i] and set label
    for i in range(0, len(a)):
        plt.fill_between(x, fa[i], alpha=0.1, linewidth=0)
        plt.annotate(labelText.replace("s1", str(a[i]))
                     + r"${}$".format(np.round(np.trapz(fa[i], x), 2)),
                     (3.1, fa[i][-1]))

    plt.xlabel(r"x")
    plt.ylabel(r"$f_{a}(x)$")
    plt.xlim(-3, 5)
    plt.xticks([-3, -2, -1, 0, 1, 2, 3])
    plt.ylim(0, 40)

    if save_path:
        plt.savefig(save_path)
    if show_figure:
        plt.show()
    pass


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """ 
    Generates,prints and saves (if desired) three subplots: f1,f2 and f1+f2
    f1(t)=0.5*cos(1/50*pi*t)
    f2(t)=0.25*(sin(pi*t)+sin(3/2*pi*t))

    Args:
        show_figure (bool, optional): If true, shows plot in new window. Defaults to False.
        save_path (str | None, optional): If set, creates (or updates) file with plot image. Defaults to None.
    """
    x = np.arange(0, 100.001, 0.001)

    f1 = 0.5*np.cos((1/50)*np.pi*x)
    f2 = 0.25*(np.sin(np.pi*x)+np.sin((3/2)*np.pi*x))
    f12 = f1+f2

    f12upper = np.ma.masked_where(f12 >= f1, f12)
    f12lower = np.ma.masked_where(f12 <= f1, f12)

    plt.close()
    fig, ax = plt.subplots(3,
                           constrained_layout=True)
    ax[0].plot(x, f1, linewidth=1.1)
    ax[1].plot(x, f2, linewidth=1.1)
    ax[2].plot(x, f12lower, color="green", linewidth=1.1)
    ax[2].plot(x, f12upper, color="red", linewidth=1.1)

    for i in ax:
        i.set_ylim(-0.8, 0.8)
        i.set_yticks([-0.8, -0.4, 0.0, 0.4, 0.8])
        i.set_xlim(0, 100)
        i.set_xticks([0, 20, 40, 60, 80, 100])
        i.set_xlabel("t")

    ax[0].set_ylabel(r"$f_{1}(t)$")
    ax[1].set_ylabel(r"$f_{2}(t)$")
    ax[2].set_ylabel(r"$f_{1}(t)+f_{2}(t)$")

    if save_path:
        plt.savefig(save_path)
    if show_figure:
        plt.show()
    pass


def download_data() -> List[Dict[str, Any]]:
    """
    Downloads webpage containing specified data, selects, formates and store this data in list of dictionaries
    Specified webpage: https://ehw.fit.vutbr.cz/izv/stanice.html
    Webpage containing data: https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html
    Webpage containing data was found with browser developer setting in specified webpage

    Returns:
        List[Dict[str, Any]]: List with Dictionaries {'position', 'lat', 'long', 'height'}
    """
    response = requests.get("https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html")

    soup = BeautifulSoup(response.content, "html.parser")

    rawRows = soup.find_all('tr', {'class': ['nezvyraznit', 'zvyraznit']})

    result = []
    for i in rawRows:
        subSoup = BeautifulSoup(str(i), "html.parser")
        elem = subSoup.find_all('td')
        line = {
            'position': elem[0].text,
            'lat': float(elem[2].text[:-1].replace(',', '.')),
            'long': float(elem[4].text[:-1].replace(',', '.')),
            'height': float(elem[6].text.strip().replace(',', '.'))
        }
        result.append(line)

    return result

if __name__ == "__main__":
    generate_graph([1,2,3],True)
    print(integrate.__doc__)
    print(generate_graph.__doc__)
    print(generate_sinus.__doc__)
    print(download_data.__doc__)
