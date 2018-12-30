import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import squarify

data = pd.read_excel("ToAnalize.xlsx")
data["engagement"] = data[['Like', 'Love','Haha', 'Angry', 'Wow', 'Sad']].sum(axis = 1)
data["engagement"] = data["engagement"]  + 5*data['Comment'] + 10*data["Share"]

data = data.loc[data.engagement < 1500]  # saco outliers
data = data.loc[data.engagement > 30]  # saco outliers

#plt.boxplot(data["engagement"])

data["engagement"] = 1000*(1-np.exp(-np.array(data["engagement"])/1000))

colores = ['celeste', 'azul', 'rosa', 'rojo', 'naranja',
                          'amarillo', 'verde', 'negro', 'blanco', 'cremita', 'gris']

# muestro como se usa la paleta de colores:
ser = data[colores].mean()
desvio = data[colores].std()

plt.scatter(colores, list(ser), s=list(20*desvio), c="red", alpha=0.4)
plt.ylabel("% de colores")
plt.title("Distribución de colores")

# discretizo faces
data.loc[data["n_faces"] >0,"n_faces"] = 1
data.loc[data["n_faces"] == 0,"n_faces"] = "No Face"
data.loc[data["n_faces"] == 1,"n_faces"] = "Face"
sns.boxplot(y = "engagement",x = "n_faces",data = data)
plt.ylabel("Engagement")
plt.title("Impacto de la aparición de caras en Engagement")

#I muestreando para que quede mejor...
aux = data.loc[data.n_faces == "No Face"].sample(n = 80).append( data.loc[data.n_faces == "Face"])  # sampleo un poco
sns.boxplot(y = "engagement",x = "n_faces",data = aux)

# Brillo:
plt.plot(data["brillo"],data["engagement"],'*')
#data.loc[data.brillo < 0.25,"hue"] = "Range0%-70%"
#data.loc[(data.brillo < 0.75) & (data.brillo >= 0.5),"hue"] = "Range72% - 85%"
#data.loc[data.brillo > 0.75,"hue"] = "Range85%-"
data.loc[data.brillo < 0.8,"hue"] = "Range0%-70%"
data.loc[data.brillo >= 0.8,"hue"] = "Range70%-"
sns.boxplot(y = "engagement",x = "hue",data = data)

# testeo ANOVA para ver que onda...

import statsmodels.api as sm
from statsmodels.formula.api import ols
 
mod = ols('engagement ~ n_faces',
                data=data).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print (aov_table)

# Saturacion
#plt.plot(data["saturacion"],data["engagement"],'*')

data.loc[data.saturacion < 0.6,"hue"] = "Range0%-60%"
data.loc[data.saturacion >= 0.6,"hue"] = "Range60%-"
sns.boxplot(y = "engagement",x = "hue",data = data)

# RAtio:

plt.plot(data.loc[data.ratio2 == 1]["ratio1"],data["engagement"].loc[data.ratio2 == 1],'*')
sns.boxplot(x = "ratio1",y = "engagement",data =  data.loc[data.ratio2 == 1])
plt.boxplot(data["engagement"])

####### labels:

from ast import literal_eval

# transformo el string lista a una lista normal!
data["labels"] = data["labels"].fillna( '[]')
data.loc[:,'labels'] = data.loc[:,'labels'].apply(lambda x: literal_eval(x))

# saco stopwords
stop_words = ["brand","advertising","and","yellow","magenta","red","blue","pink","purple","text","violet","photo",
              "graphics","line","graphic","design","font","wallpaper","banner","area","illustration","film",
              "poster","pattern","album","cover","sized","to","caption","human"]
lemmatizer = {'happiness':'happy','fun':'happy','smile':'happy','cats':'cat','whiskers':'cat','vision':'eyewear',
              'electronic':'technology','phone':'mobile','glasses':'eyewear','dessert':'food','baking':'food',
              'appetizer':'food',"breakfast":"food"}

data.loc[:,'labels'] = data.loc[:,'labels'].apply(lambda x: [y for y in x if(y not in stop_words)]) # stop words
data.loc[:,'labels'] = data.loc[:,'labels'].apply(lambda x:  [lemmatizer.get(y,y) for y in x ]) # lematize
data.loc[:,'labels'] = data.loc[:,'labels'].apply(lambda x: list(set(x))) # drop duplis per row

df_words_engagement = pd.DataFrame([],columns = ["word","engagement"])
for ind,line in data.iterrows():
    # me creo un dataframe con todas las palabras y engagement
    aux = pd.DataFrame([],columns = ["word","engagement"])
    aux["word"] = line["labels"]
    aux["engagement"] = line["engagement"]
    df_words_engagement = df_words_engagement.append(aux)

summerized = pd.DataFrame(df_words_engagement.groupby("word")["engagement"].median())
summerized["count"] = pd.DataFrame(df_words_engagement.groupby("word")["engagement"].count())
summerized.columns = ["mean","count"]

summerized = summerized.loc[summerized["count"] > 3]  # umbralizar segun corresponda!
summerized["word"] = list(summerized.index.values)

#♣ Treemap

# create a color palette, mapped to these values

my_values = summerized['mean']/10
cmap = matplotlib.cm.summer
mini=min(my_values)
maxi=max(my_values)
norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in my_values]
 
squarify.plot(sizes=summerized['mean']/10, label=summerized['word'], alpha=.6, color=colors  )
plt.axis('off')
plt.show()

### Find labels
def finder_label(df,x):
    l = []
    for ind,line in df.iterrows():
        if(x in line["labels"]):
            l.append(True)
        else:
            l.append(False)
    
    return df.loc[l]

cond = finder_label(data,'behavior')

### Height
#plt.plot(data["height"],data["engagement"],'*')
height = [np.mean(data.loc[data.height < 750]["engagement"]),np.mean(data.loc[data.height >= 750]["engagement"])]
bars = ["Range < 900","Range > 900"]
y_pos = np.arange(len(bars)) 
plt.bar(y_pos, height) 
plt.xticks(y_pos, bars)  
plt.ylabel("Engagement")
plt.xlabel("Height")
plt.title("Impacto de la altura en Engagement")

### Width
#plt.plot(data["width"],data["engagement"],'*')
height = [np.mean(data.loc[data.width < 1000]["engagement"]),np.mean(data.loc[data.width >= 1000]["engagement"])]
bars = ["Range < 1000","Range > 1000"]
y_pos = np.arange(len(bars)) 
plt.bar(y_pos, height) 
plt.xticks(y_pos, bars)  
plt.ylabel("Engagement")
plt.xlabel("Ancho de la imagen")
plt.title("Impacto del Ancho en Engagement")

## Texto
plt.plot(data["num_text_leters"],data["engagement"],'*')

data.loc[data.num_text_leters < 80,"hue"] = "Range0%-70%"
data.loc[data.num_text_leters >= 80,"hue"] = "Range70%-"
sns.boxplot(y = "engagement",x = "hue",data = data)

# Ismarca
data.loc[data["ismarca"] >0,"ismarca"] = 1
height = [np.mean(data.loc[data.ismarca == 0]["engagement"]),np.mean(data.loc[data.ismarca == 1]["engagement"])]
bars = ["Menciona marca","NO menciona marca"]
y_pos = np.arange(len(bars)) 
plt.bar(y_pos, height) 
plt.xticks(y_pos, bars)  
plt.ylabel("Engagement")
plt.xlabel("Menciona?")
plt.title("Mención de la marca en creativo vs Engagement")

from LDA_preproc import LDA_preproc

data_study = data.copy()
data_study = data_study.dropna(subset = ["Texte"])
data_study["Full Text"] = data_study["Texte"]
preproc = LDA_preproc(data_study)
preproc.preprocessing()  # arranco el preprocasamiento!   

new_stopwords = pd.read_excel("StopWords2.xlsx")['palabras'].tolist() 
preproc.update_StopWords(new_stopwords )

Counter_df = preproc.countVectorizer()

data_study["labels"] = preproc.get_procTokenTweets()
data_study['labels'] = data_study['labels'].apply(lambda x: list(set(x))) # drop duplis per row

df_words_engagement = pd.DataFrame([],columns = ["word","engagement"])
for ind,line in data_study.iterrows():
    # me creo un dataframe con todas las palabras y engagement
    aux = pd.DataFrame([],columns = ["word","engagement"])
    aux["word"] = line["labels"]
    aux["engagement"] = float(line["engagement"])
    df_words_engagement = df_words_engagement.append(aux)

summerized = pd.DataFrame(df_words_engagement.groupby("word")["engagement"].median())
summerized["count"] = pd.DataFrame(df_words_engagement.groupby("word")["engagement"].count())
summerized.columns = ["mean","count"]

summerized = summerized.loc[summerized["count"] > 4]  # umbralizar segun corresponda!
summerized["word"] = list(summerized.index.values)

#summerized["mean"] = summerized["mean"]*summerized["mean"]   # cuadrado
cond = finder_label(data_study,'checo')

summerized.to_excel(".2Tableau.xlsx",index = False)# plot it in Tableau





