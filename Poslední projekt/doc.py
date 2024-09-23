import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_pickle("accidents.pkl.gz")

# collect only valid data
df=df.dropna(subset=["p11","region","p2a","p13a","p13b","p13c"])
df=df[(df["p11"] > 0)&(df["p11"]<10)&(df["p2a"]>="2020-01-01")]
total_accidents=df.shape[0]

# determine if record included injury or drugs have been present
df["under_influence"]=np.where(df["p11"]!=2,True,False)
drunk_accidents=df["under_influence"].value_counts().get(True, 0)
sober_accidents=df.shape[0]-drunk_accidents
df["injury"]=np.where((df["p13a"]!=0)|(df["p13b"]!=0)|(df["p13c"]!=0),True,False)

# agregate records based on col "under_influence"
df=df.groupby(["region","injury","under_influence"])["under_influence"].value_counts()
df=df.reset_index(name="count")

# plot records including injury and drugs
sns.set_style("darkgrid")
sns.catplot(x="region", y="count", data=df, kind="bar")
g = sns.catplot(x="region", y="count",
    data=df[(df["injury"]==True)&(df["under_influence"]==1)],kind="bar",hue="region",height=4, aspect=2)
g.fig.subplots_adjust(hspace=0.4, wspace=0.3,top=0.9)
ax=g.axes.flat[0]
ax.set_ylabel("Počet nehod",fontsize=14)
ax.set_xlabel("Kraj",fontsize=14)
ax.set_title("Nehody se zraněním či úmrtím v jednotlivých krajích pod vlivem:",fontsize=16)
plt.savefig("fig.png")

# filter records without injury, drop col "injury", pivot table
df=df[df["injury"]==True].drop("injury",axis=1)
df = df.pivot_table(index="region", columns="under_influence", values="count", aggfunc="sum", fill_value=0)
df.columns = ["sober","under influence" ]
df.reset_index(inplace=True)

# count injuries, print dataframe
sober_injuries=df["sober"].sum()
drunk_injuries=df["under influence"].sum()
df=df.rename(columns={"region":"Kraj","under influence":"Pod vlivem","sober":"Střízliví"})
df_csv = df.to_csv(index=False)
print(df_csv)

# print data used in essay
print("Celkem nehod: "+str(total_accidents))
print("Střízliví řidiči: ")
print("\tPočet nehod: "+str(sober_accidents))
print("\tPočet nehod se zraněním či úmrtím: "+str(sober_injuries))
print("\tPoměr nehod se zraněním ku počtu nehod: {:.2f}%".format(sober_injuries*100/sober_accidents))
print("Opilí řidiči:")
print("\tPočet nehod: "+str(drunk_accidents))
print("\tPočet nehod se zraněním či úmrtím: "+str(drunk_injuries))
print("\tPoměr nehod se zraněním ku počtu nehod: {:.2f}%".format(drunk_injuries*100/drunk_accidents))

