#%%
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10
from typing import Literal, Dict, List
import torch
import matplotlib.pyplot as plt

working_dir = Path("./results")
data_name = 'CDSBART_BEST'
assert working_dir.is_dir, f"Working directory {working_dir} does not exist."

i=3
data_info = pd.read_csv(working_dir / f"{data_name}_{i}_df.csv")


data_info
# data_info = data_info[data_info['y'].isin([0,1,2,3])]
print(set(data_info['y']))


class VizData:
    def __init__(
        self,
        working_dir: str | Path = working_dir,
        info_df: pd.DataFrame = data_info,
        choice_embedding_data: Literal["with_eos", "no_eos"] = "no_eos",
        choice_for_embedding: Literal["mean", "max", "eos"] = "mean", 
        label_info: Dict = {'plant':0 ,'bacteria':1,'invertebrate':2,'fungi':3} #, 'vertebrate_other':4},
       ):

        assert working_dir.is_dir(), f"Working directory {working_dir} does not exist."

        self.info = info_df
        self.labels = info_df["y"].values
        self.label_info = label_info
        self.reverse_label_info = {v: k for k, v in label_info.items()}

        self.choice_embedding_data = choice_embedding_data
        self.choice_for_embedding = choice_for_embedding

        self.umap_df = pd.DataFrame()
        self.umap_df["labels"] = self.labels

        # use bokeh color palette
        # self.palette = Category10[10]
        # choose 5 colors for the 5 classes

        # use user def colors
        colors = ['darkorange', 'navy', 'green', 'red'] #, 'red', 'green', 'brown','pink','yellow']
        markers = ['circle', 'square', 'triangle', 'diamond']
        # colors = ['darkorange', 'navy', 'green', 'red', 'pink'] #, 'red', 'green', 'brown','pink','yellow']
        # markers = ['circle', 'square', 'triangle', 'diamond','cross']
        
        sizes = [10, 10, 12, 14]
        # sizes = [8, 8, 10, 12, 10]
        self.umap_df['color'] = self.umap_df['labels'].map(lambda x: colors[x])
        self.umap_df['marker'] = self.umap_df['labels'].map(lambda x: markers[x])
        self.umap_df['legend'] = self.umap_df['labels'].map(self.reverse_label_info)
        self.umap_df['size'] = self.umap_df['labels'].map(lambda x: sizes[x])
        
        self.embedding_data = self.get_embedding_data(choice_embedding_data)
        print(self.embedding_data)
        self.embeddings = self.get_embedding(choice_for_embedding, self.embedding_data)

        print(f"Loaded {self.choice_embedding_data} embedding with shape {self.embeddings.shape}.")

    @staticmethod
    def get_embedding_data(choice):
        match choice:
            case "with_eos":
                embedding_data = torch.load(working_dir / f"emb_with_eos_{data_name}_{i}.pt")
            case "no_eos":
                embedding_data = torch.load(working_dir / f"emb_no_eos_{data_name}_{i}.pt") 
                # embedding_data = torch.load("state_embeddings_no_eos.pt").to("cpu") 

            case _:
                raise ValueError("Invalid choice for embedding data. Choose 'with_eos' or 'no_eos'.")
            
        return embedding_data 

    @staticmethod
    def get_embedding(choice, embedding_data):
        match choice:
            case "mean":
                embeddings = embedding_data.mean(dim=1)
            case "max":
                embeddings = embedding_data.max(dim=1) 

            case _:


                raise ValueError("Invalid choice for embedding. Choose 'mean', 'max', or 'eos'.")
        return embeddings
    
viz = VizData(
    working_dir=working_dir,
    info_df=data_info,
    choice_embedding_data="no_eos",
    choice_for_embedding="mean",
)

#%%
pca = PCA(n_components=50)
pca_embeddings = pca.fit_transform(viz.embeddings)


umap_model = umap.UMAP(
    n_neighbors=20,
    n_epochs=1500,
    min_dist=0.0,
    init='spectral',
    repulsion_strength=4,
    # metric='euclidean',
    metric = 'manhattan',
    # n_components=2,
    random_state=42,
# spread=0.3
)


# umap_embeddings = umap_model.fit_transform(viz.embeddings)
umap_embeddings = umap_model.fit_transform(pca_embeddings)
umap_df = viz.umap_df

# add umap embeddings to the dataframe
umap_df["x"] = umap_embeddings[:, 0]
umap_df["y"] = umap_embeddings[:, 1]
# Map colors to labels

plt.figure(figsize=(10, 8))
# 클러스터별 마커 설정
umap_df['legend'] = umap_df['legend'].map({'plant':'Plant','bacteria':'Bacteria',\
                          'invertebrate':'Invertebrate', 'fungi':'Fungi'})#,'vertebrate_other':'Vertebrate (other)'})
markers_palette = {
    'Plant':'o', 
    'Bacteria':'s', 
    'Invertebrate' :'D', 
    'Fungi' :'^',
    #'Vertebrate (other)' : 'v',
    }
unique_labels = umap_df['labels'].unique()

# 플롯 생성
plt.figure(figsize=(10, 8))
# 사용자 정의 색상 팔레트
custom_palette = {
    'Plant': 'red',
    'Bacteria': 'purple',
    'Invertebrate': 'navy',
    'Fungi': 'green',
   # 'Vertebrate (other)': 'darkorange'
}
size_palette = {
    'Plant': 30,
    'Bacteria': 30,
    'Invertebrate': 30,
    'Fungi': 40,
   # 'Vertebrate (other)': 40
}
print(umap_df)
# 각 클러스터에 대해 개별적으로 scatter plot 생성
for label in custom_palette.keys():
    subset = umap_df[umap_df['legend'] == label]
    plt.scatter(subset['x'], subset['y'], 
                label=label, 
                color=custom_palette[label], 
                marker=markers_palette[label], 
                alpha=0.6, 
                s = size_palette[label], 
                edgecolor='white', 
                linewidth=1.5,
                )


plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(prop={'size':13,'weight':'bold'},frameon=False)

# x축과 y축의 눈금 없애기
plt.xticks([])  # x축 눈금 없애기
plt.yticks([])  # y축 눈금 없애기

# 그래프 테두리 설정
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

filename = f'./Embedding_space_data/embedding_matplot_{i}.png'
plt.savefig(filename, dpi=300)  # 해상도를 300 DPI로 설정하여 저장

# 플롯 표시
plt.show()

# %%
