#!/usr/bin/env python3.11
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile
from io import BytesIO

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

# Ukol 1: nacteni dat ze ZIP souboru


def load_data(filename: str) -> pd.DataFrame:
    """
    Loads data from zip architecture and stores them
    in DataFrame with new region column
    
    Args:
        filename (str): path to parent zip file
        
    Returns: 
        DataFrame: dataframe with loaded data
    """
    # tyto konstanty nemente, pomuzou vam pri nacitani
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
               "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28",
               "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
               "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "p5a"]

    # def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }
    dtypes = {"a": str, "b": str, "l": str, "n": str, "o": str}
    with zipfile.ZipFile(filename) as zip_file:
        
        df_list = []
        
        for year_zip in zip_file.namelist():
            
            with zip_file.open(year_zip) as file:
                with zipfile.ZipFile(BytesIO(file.read())) as inner_zip:
                    
                    for csv_file in inner_zip.namelist():
                        
                        region_type = csv_file[:2]
                        
                        with inner_zip.open(csv_file) as f:
                            region = None
                            for key, val in regions.items():
                                # store only desired regions
                                if val == region_type:
                                    region = key
                            if region != None:
                                tmp_df = pd.read_csv(
                                    f, names=headers, encoding="cp1250", delimiter=";", dtype=dtypes)
                                tmp_df["region"] = region
                                df_list.append(tmp_df)
    df = pd.concat(df_list, ignore_index=True)
    return df
# # Ukol 2: zpracovani dat

def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Reads dataframe, removes empty rows and categorizes cols
    
    Args:
        pd (DataFrame): DataFrame with data
        verbose(bool): Print with memory usage
        
    Returns: 
        DataFrame: dataframe with parsed data
    """
    # cols with wild range of values (except floats), others are easy to cathegorizes
    unoptimized_cathegories=["region","p2","p2a","p2b",
                             "p13a","p13b","p13c","p33d",
                             "p33e","p53","p33f","p33g",
                             "d","e","p2a"]
    
    new_df = df.copy()
    new_df["date"] = pd.to_datetime(new_df["p2a"])
    new_df = new_df.drop_duplicates(subset="p1")

    # each non float and not unoptimized shall be cathegorized
    for col in new_df.columns:
        if col not in unoptimized_cathegories and new_df[col].dtype != "float64":
            new_df[col] = new_df[col].astype("category")

    # set all float colls proper data type
    decimal_cols = new_df.select_dtypes(include=["float64", "float"]).columns
    new_df[decimal_cols] = new_df[decimal_cols].astype(float).round(1)
    
    if verbose:
        original_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
        new_memory = new_df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"orig_size={original_memory:.1f} MB")
        print(f"new_size={new_memory:.1f} MB")
    return new_df

# Ukol 3: počty nehod oidke stavu řidiče

def plot_state(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    """
    Plots how many accidents happend in each region for
    various status
    
    Args:
        pd (DataFrame): DataFrame with data
        fig_location (str): Path to saved png
        show_figure (bool): Decides whetever plot should show up
        
    """
    # state dictionary for labels
    state_str={
        1:"dobrý",
        2:"unaven, usnul, náhlá fyzická indispozice",
        3:"pod vlivem léků, narkotik",
        4:"pod vlivem alkoholu, obsah alkoholu v krvi do 0,99 ‰",
        5:"pod vlivemalkoholu, obsah alkoholu v krvi 1 ‰ a více",
        6:"nemoc nebo úraz",
        7:"invalida",
        8:"řidič při jízdě zemřel (infarkt apod.)",
        9:"sebevražda",
        }
    
    # states about to be plotted
    shown_states=[state_str[2],state_str[3],state_str[7],state_str[6],state_str[9],state_str[8]]
    new_df = df.copy()
    new_df["status"] = new_df["p57"].map(state_str)
    
    new_df = new_df.groupby("region")["status"].value_counts()
    new_df = new_df.reset_index(name="count")
    new_df = new_df[new_df["status"].isin(shown_states)]
    
    sns.set_style("darkgrid")
    
    g = sns.catplot(x="region", y="count",
        col="status",data=new_df,
        kind="bar",aspect=1.2,
        col_wrap=2,legend=False,
        hue="region",sharex=False,
        sharey=False)
    
    g.fig.subplots_adjust(hspace=0.4, wspace=0.3,top=0.9)
    g.fig.suptitle("Počet nehod dle stavu", fontsize=16)  
    
    # set labels and subplot title
    for ax, state in zip(g.axes.flat, shown_states):
        ax.set_ylabel("Počet Nehod")
        ax.set_xlabel("Kraj")
        ax.set_title("Status: "+state)
        ax.tick_params(labelleft=True)

    if fig_location:
        plt.savefig(fig_location)
        
    if show_figure:
        plt.show()
    

# Ukol4: alkohol v jednotlivých hodinách


def plot_alcohol(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Plots how many accidents happend because of alcohol or
    other drugs each hour in contrast to sober accidents
    
    Args:
        pd (DataFrame): DataFrame with data
        fig_location (str): Path to saved png
        show_figure (bool): Decides whetever plot should show up
        
    """
    # shown regions with their label variant
    my_regions=["PHA","JHM","OLK","ZLK"]
    my_regions_full=["Praha","Jihomoravský","Olomoucký","Zlínský"]
    new_df = df.copy()
    
    new_df = new_df[new_df["region"].isin(my_regions)]
    new_df = new_df.dropna(subset=["p2b"])
    new_df = new_df[new_df["p2b"] <= 2359]
    new_df["p2b"] = np.floor(new_df["p2b"] / 100).astype(int)
    new_df["p11"] = new_df["p11"].cat.codes
    new_df["sobriety"] = np.where(new_df["p11"].isin([1, 2]), True, new_df["p11"] <= 3)
    
    sns.set_style("darkgrid")
    g = sns.FacetGrid(new_df, col="region", col_wrap=2,aspect=2, sharex=False, sharey=True)
    g.map_dataframe(sns.countplot, "p2b", hue="sobriety",palette="dark", order=sorted(new_df["p2b"].unique()))
    g.add_legend(title="Sobriety")
    g.fig.suptitle("Přítomnost alkoholu v krvi", fontsize=16) 
        
    plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)
    
    # set labels and subplot title
    for ax, region in zip(g.axes.flat, my_regions_full):
        ax.set_ylabel("Počet nehod")
        ax.set_xlabel("Hodina")
        ax.set_title("Kraj: "+region)
        ax.tick_params(labelleft=True)
    
    
    if fig_location:
        plt.savefig(fig_location)
        
    if show_figure:
        plt.show()

# Ukol 5: Zavinění nehody v čase

def plot_fault(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    """
    Plots causes of accidents each month between 1.1.2016 and 1.1.2023
    
    Args:
        pd (DataFrame): DataFrame with data
        fig_location (str): Path to saved png
        show_figure (bool): Decides whetever plot should show up
        
    """
    my_regions=["PHA","JHM","OLK","ZLK"]
    my_regions_full=["Jihomoravský","Olomoucký","Praha","Zlínský"]
    xticks=["01/16","01/17","01/18","01/19",
            "01/20","01/21","01/22","01/23",]
    
    fault_dict={
        1:"motorové v.",
        2:"nemotorové v.",
        3:"Chodec",
        4:"Zvířetem",
    }
    new_df = df.copy()
    
    new_df = new_df[new_df["region"].isin(my_regions)& new_df["p10"].isin(range(1,5))]
    new_df.loc[:, "date"] = new_df["date"].dt.to_period("M")
    
    # pivoted dataframe
    new_df = new_df.pivot_table(index=["date", "region"], columns="p10", aggfunc="size", fill_value=0).drop(columns=[0,5,6,7])
    new_df = new_df.stack().reset_index()
    new_df.columns = ["date", "region", "accident", "count"]

    # stacked dataframe
    new_df["date"] = new_df["date"].dt.strftime("%m/%y")
    new_df["accident"] = new_df["accident"].astype(str)

    sns.set_style("darkgrid")
    g = sns.FacetGrid(new_df, col="region",col_wrap=2,aspect=2, sharey=False,sharex=False)
    g.map_dataframe(sns.lineplot, x="date", y="count", hue="accident")
    g.fig.suptitle("Příčiny nehod", fontsize=16) 
    
    # set labels and subplot title
    for ax, region in zip(g.axes.flat, my_regions_full):
        ax.set_ylabel("Počet nehod")
        ax.set_xlabel("")
        ax.set_title("Kraj: "+region)
        ax.tick_params(labelleft=True)
        ax.set_xticks(xticks)
    
    g.add_legend(title="Viník", loc="center right")
    g.fig.subplots_adjust(right=0.85,top=0.9,hspace=0.4, wspace=0.3)
    
    # set legend labels
    for t, l in zip(g._legend.texts, fault_dict.values()):
        t.set_text(l)
    
    if fig_location:
        plt.savefig(fig_location)
        
    if show_figure:
        plt.show()


import time
if __name__ == "__main__":
    start_time = time.time()
    data=parse_data(load_data("./data/data.zip"),True)
    plot_alcohol(data,fig_location="alcohol.png")
    plot_state(data,fig_location="state.png")
    plot_fault(data,fig_location="faults.png")
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Trvalo mi to: {round(elapsed_time, 2)} sekund :-)")
    pass


# # Poznamka:
# # pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# # VS Code a oznaceni jako bunky (radek #%%% )
# # Pak muzete data jednou nacist a dale ladit jednotlive funkce
# # Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# # ucely) a nacitat jej naparsovany z disku


