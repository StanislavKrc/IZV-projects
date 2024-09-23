#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily
import sklearn.cluster
import numpy as np
# muzete pridat vlastni knihovny
from mpl_toolkits.axes_grid1 import make_axes_locatable


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani"""
    
    # for out of bounds EPSG coordinates removal
    x_boundary=[-159365.31,-911053.67]
    y_boundary=[-951499.37, -1353292.51]
    filtered_df = df.dropna(subset=["d","e"])
    filtered_df=filtered_df[(filtered_df["d"]<=x_boundary[0]) &
                            (filtered_df["d"]>=x_boundary[1]) &
                            (filtered_df["e"]<=y_boundary[0]) &
                            (filtered_df["e"]>=y_boundary[1])]
    
    gdf=geopandas.GeoDataFrame(filtered_df,geometry=geopandas.points_from_xy(filtered_df["d"],filtered_df["e"]),crs="EPSG:5514")
    return gdf

def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu s nehodami  """
    
    local_gdf=gdf[(gdf["date"]>="2021-01-01") & (gdf["date"]<"2023-01-01") & (gdf["region"]=="JHM") & ((gdf["p10"]==4))]
    
    _ , ax = plt.subplots(1, 2, figsize=(20, 7),constrained_layout=True)
    
    gdf_21=local_gdf[(local_gdf["date"]>="2021-01-01") & (local_gdf["date"]<"2022-01-01")]
    gdf_22=local_gdf[(local_gdf["date"]>="2022-01-01") & (local_gdf["date"]<"2023-01-01")]
    
    gdf_21.plot(ax=ax[0],color="red",markersize=15)
    gdf_22.plot(ax=ax[1],color="red",markersize=15)
    
    for axis in ax:
        axis.set_xlim(local_gdf.total_bounds[0]-10000,local_gdf.total_bounds[2]+10000)
        axis.set_ylim(local_gdf.total_bounds[1]-10000,local_gdf.total_bounds[3]+10000)
        axis.set_axis_off()

    plt.suptitle("Nehody zaviněné zvěří v letech:",fontsize=24,weight="bold")
    ax[0].set_title("2021",fontsize=16)
    ax[1].set_title("2022",fontsize=16)

    contextily.add_basemap(ax[0], crs=local_gdf.crs.to_string(), url=contextily.providers.CartoDB.Voyager)
    contextily.add_basemap(ax[1], crs=local_gdf.crs.to_string(), url=contextily.providers.CartoDB.Voyager)
    
    if fig_location:
        plt.savefig(fig_location)
        
    if show_figure:
        plt.show()
    pass

def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """
    
    local_gdf = gdf[(gdf["region"] == "JHM") & (gdf["p11"] >= 4)]
    
    points = np.array(local_gdf["geometry"].apply(lambda point: (point.x, point.y)).tolist())
    
    clusters=12
    
    kmeans = sklearn.cluster.KMeans(n_clusters=clusters, random_state=0,n_init=10).fit(points)
    
    local_gdf = local_gdf.copy()
    local_gdf["cluster"] = kmeans.labels_
    
    clustered_data = local_gdf.groupby("cluster")["geometry"].apply(lambda x: x.unary_union.convex_hull)
    cluster_counts = local_gdf["cluster"].value_counts()

    norm = plt.Normalize(cluster_counts.min(), cluster_counts.max())
    cmap = plt.get_cmap("viridis")
    
    _, ax = plt.subplots(figsize=(15, 11))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, cax=cax,orientation="horizontal")
    cbar.set_label( label="Četnost nehod",fontsize=16)

    clustered_data.plot(ax=ax, facecolor="grey",alpha=0.4, lw=2, zorder=1)
    
    for cluster in range(clusters):
        data = local_gdf[local_gdf["cluster"] == cluster]
        color = cmap(norm(cluster_counts[cluster]))
        data.plot(ax=ax, markersize=20, color=color, label=f"Cluster {cluster}",zorder=2)
        

    contextily.add_basemap(ax, crs=local_gdf.crs.to_string(), url=contextily.providers.CartoDB.Voyager)
    plt.suptitle("Nehody zaviněné alkoholem či drogami:",fontsize=24,weight="bold")
    
    ax.set_axis_off()
    
    if fig_location:
        plt.savefig(fig_location)
        
    if show_figure:
        plt.show()
    pass

if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", False)
    plot_cluster(gdf, "geo2.png", False)
