import os
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

DIR_PATH = "orl_faces/s"
FIGURE_SIZE = (10,10)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="Insira o caminho da imagem em orl_faces Ex.: s2/6.pgm")
args = parser.parse_args()

image = None
if args.image:
    image = args.image
else:
    print("Nenhuma imagem informada")
    exit(1)


X = []
y = []
X_testing = []
y_testing = []
test_image = None

# Carregamos as imagens e montamos quatro conjuntos de dados: 
# um para features e um para labels e outros dois também com features e labels, mas excluindo a imagem de teste
for i in range(1, 41):
    im_group = DIR_PATH + str(i)
    for entry in os.scandir(im_group):
        img = cv.imread(entry, cv.IMREAD_GRAYSCALE)
        
        # Resize para (112x92), talvez seja boa ideia também aplicar scaling para manter o aspecto
        img = cv.resize(img, (112, 92), interpolation=cv.INTER_AREA)
        img_flat = img.ravel()

        if f"s{i}/{entry.name}" == image:
            test_image = img_flat
            test_image_original = img
        else:
            X_testing.append(img_flat)
            y_testing.append(i - 1)
            
        X.append(img_flat)
        y.append(i - 1)
        
    print(f"GRUPO {i} CARREGADO.")

X = np.array(X, dtype=np.uint8)
y = np.array(y, dtype=np.uint8)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(50)
X_pca_tsne = pca.fit_transform(X_scaled)

tsne = TSNE(n_components=2)
projection = tsne.fit_transform(X_pca_tsne)

plt.figure(figsize=FIGURE_SIZE)
plt.scatter(projection[:, 0], projection[:, 1], c=y, cmap=plt.cm.rainbow)
plt.title("t-SNE")
plt.savefig("figures/tsne")


kfold = StratifiedKFold(5, shuffle=True, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(50)),
    ("svm", SVC(kernel="rbf", random_state=42))
])

validator = cross_val_predict(pipeline, X, y, cv=kfold, n_jobs=-1, verbose=True)
cf_matrix = confusion_matrix(y, validator)

plt.figure(figsize=FIGURE_SIZE)
sns.heatmap(cf_matrix, cmap="Blues", cbar=True, square=True)
plt.title("Matriz de confusão 5-Fold")
plt.xlabel("Classe prevista")
plt.ylabel("Classe verdadeira")
plt.savefig("figures/confusion_matrix.png")

# Agora treinamos outro SVM sem incluir a imagem a ser prevista nos dados
scaler = StandardScaler()
pca = PCA(50)

X_testing = scaler.fit_transform(X_testing)
X_testing = pca.fit_transform(X_testing)

# Aplicados z-score e pca na imagem de teste também
test_image = test_image.reshape(1, -1)
test_image = scaler.transform(test_image)
test_image = pca.transform(test_image)

svm2 = SVC(kernel="rbf", probability=True, random_state=42)
svm2.fit(X_testing, y_testing)
prediction = svm2.predict(test_image)[0]
proba = svm2.predict_proba(test_image)

sort_indices = np.argsort(proba[0])[::-1]
sort_proba = proba[0][sort_indices]
print(f"Predição: {prediction + 1}")

for i in range(5):
    print(f"Prob classe {sort_indices[i] + 1}: {sort_proba[i] * 100:.2f}%")


class_dir = os.path.join(DIR_PATH + str(prediction + 1))

images_same_class = []
for entry in os.scandir(class_dir):
    if f"s{prediction + 1}/{entry.name}" == image:
        continue
    img = cv.imread(entry.path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (112, 92), interpolation=cv.INTER_AREA)
    images_same_class.append(img)

plt.figure(figsize=(4, 4))
plt.imshow(test_image_original, cmap='gray')
plt.title(f"Imagem de Teste\nClasse Prevista: s{prediction + 1}")
plt.axis('off')
plt.tight_layout()

image_label = image.split("/")
image_label = f"{image_label[0]}_{image_label[1]}"
plt.savefig(f"figures/{image_label}_test.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.suptitle(f"Imagens da Classe Prevista: s{prediction + 1}", fontsize=14)

for i in range(min(9, len(images_same_class))):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images_same_class[i], cmap='gray')
    plt.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"figures/{image_label}_predicted_class.png")
plt.close()
