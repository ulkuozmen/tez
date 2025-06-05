#!/usr/bin/env python
# coding: utf-8

# Geleneksel: TF-IDF / CountVectorizer / Hashing Vectorizer
# 
# Derin Öğrenme: BERT vektörleri
# 
# Klasik ML Modelleri:
# → Naive Bayes
# → Random Forest
# → Logistic Regression
# → SVM
# 
# Derin Öğrenme Modelleri:
# → BERT + LSTM
# → ANN (Artificial Neural Network)
# 
# 	Karşılaştırma: Accuracy / Precision / Recall / F1 Score ile tüm modelleri karşılaştırmak

# In[17]:


import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE


# In[18]:


# Veri Setini Yükleme
file_path = "phishing_site_urls.csv"  # Dosyanın bulunduğu yolu belirtin
df = pd.read_csv(file_path)


# In[19]:


df.info()


# In[20]:


print("\n Veri seti boyutu:")
print(df.shape)


# In[21]:


# Eksik değer kontrolü
print("\n Eksik Değer Sayısı:")
print(df.isnull().sum())


# In[22]:


# Etiket (Label) dağılımı
print("\n Etiket Dağılımı:")
print(df['Label'].value_counts())


# In[23]:


# Yinelenen satır sayısını bul
yinelenen_sayi = df.duplicated().sum()
print(f" Yinelenen kayıt sayısı: {yinelenen_sayi}")


# In[24]:


df = df.drop_duplicates()
print(f" Yinelenen satırlar temizlendi. Yeni veri seti boyutu: {df.shape}")


# In[25]:


print("Etiket dağılımı (Label):")
print(df['Label'].value_counts())


# In[26]:


# Kategorik Değerlerin İşlenmesi
df['Label'] = df['Label'].map({'good': 1, 'bad': 0})

print(df.head())


# In[27]:


# Etiket dağılımı
plt.figure(figsize=(6,4))
sns.countplot(x=df['Label'], palette=['red', 'green'])
plt.title('Etiket Dağılımı ("Good" ve "Bad" URL\'ler)')
plt.xlabel('Etiket')
plt.ylabel('Adet')
plt.xticks([0, 1], ["Bad", "Good"])
plt.show()


# In[28]:


X = df['URL']  # URL'ler
y = df['Label']  # Etiketler


# In[29]:


#  Eğitim-Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

# Vektörizerler
vectorizers = {
    "CountVectorizer": CountVectorizer(stop_words='english', max_features=5000),
    "TF-IDF": TfidfVectorizer(stop_words='english', max_features=5000),
    "HashingVectorizer": HashingVectorizer(n_features=5000, alternate_sign=False)
}


# In[31]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Modeller
models = {
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
   # "SVM": SVC(kernel='linear', random_state=42)
    "KNN": KNeighborsClassifier(n_neighbors=5)  #varsayılan 5 komşu
}
# Sonuçları Saklayacağımız Liste
results = []


# In[32]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

for vect_name, vect in vectorizers.items():
    print(f"\n🔵 Vektörizer: {vect_name}")

    # Eğitim ve Test Vektörlerini çıkar
    X_train_vect = vect.fit_transform(X_train)
    X_test_vect = vect.transform(X_test)

    # SMOTE uygulaması
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_vect, y_train)

    # MODELLER
    for model_name, model in models.items():
        print(f"  🔸 Model: {model_name}")

        # Modeli Eğit
        model.fit(X_train_smote, y_train_smote)

        # Test verisi üzerinde tahmin
        y_pred = model.predict(X_test_vect)

        # Performans Metrikleri
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Sonuçları kaydet
        results.append({
            "Vectorizer": vect_name,
            "Model": model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })


# In[33]:


print("SMOTE sonrası eğitim veri seti boyutu:", X_train_smote.shape)
print("SMOTE sonrası etiket dağılımı:", np.bincount(y_train_smote))


# In[34]:


from tabulate import tabulate
results_df = pd.DataFrame(results)

print("\n✅ Tüm Sonuçlar:")
print(tabulate(results_df, headers='keys', tablefmt='pretty'))


# In[36]:


# En iyi sonucu bul
best_result = sorted_results.iloc[0]

print("\n🏆 En İyi Model:")
print(f"Vektörizer: {best_result['Vectorizer']}")
print(f"Model: {best_result['Model']}")
print(f"Accuracy: {best_result['Accuracy']:.4f}")
print(f"F1-Score: {best_result['F1-Score']:.4f}")


# In[66]:


import joblib

# En iyi vektörizer ve model nesnesini kaydet 
joblib.dump(vect, 'hashing_vectorizer.pkl')
joblib.dump(model, 'random_forest_model.pkl')


# In[38]:


# 1. Sonuçları sıralama
sorted_results = results_df.sort_values(by="Accuracy", ascending=False)

# 2. En iyi model
best_result = sorted_results.iloc[0]

print("\n🏆 En İyi Model:")
print(f"Vektörizer: {best_result['Vectorizer']}")
print(f"Model: {best_result['Model']}")
print(f"Accuracy: {best_result['Accuracy']:.4f}")
print(f"F1-Score: {best_result['F1-Score']:.4f}")

# 3. Accuracy'leri bar chart ile gösterelim
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.barh(
    sorted_results['Vectorizer'] + " + " + sorted_results['Model'],
    sorted_results['Accuracy'],
    color='skyblue'
)
plt.xlabel('Accuracy')
plt.title('Model Başarı Karşılaştırması')
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.show()


# In[43]:


get_ipython().system('pip install -q sentence-transformers')
from sentence_transformers import SentenceTransformer
# BERT modeli yükle
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# URL'leri küçük harfe çevir (temizleme)
X = df['URL'].str.lower()

# BERT ile embedding çıkar
X_embeddings = bert_model.encode(X.tolist(), show_progress_bar=True)

# Sonuçlar: NumPy Array
print("BERT embedding boyutu:", X_embeddings.shape)


# BERT özelliği çıkarıldı, metinler artık sayısal vektör oldu.

# In[46]:


from sklearn.model_selection import train_test_split

y = df['Label'].values  # 0 = bad, 1 = good

X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Eğitim seti boyutu:", X_train.shape)
print("Test seti boyutu:", X_test.shape)



# In[47]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(256, activation='relu', input_shape=(384,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ANN modelin giriş katmanında input_shape=(384,) yazıyor.
# 
# Bu 384 boyut, BERT modelinden çıkan embedding boyutu.
# 
# Yani doğrudan BERT'in ürettiği embeddingler ANN girişine verildi.

# In[48]:


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

BERT embeddingleri üstünde SMOTE yaparsak model anlamlı yerlerde değil, bozulmuş yapay yerlerde öğrenir.

monitor='val_loss'	Her epoch sonunda Validation Loss değerini takip eder.
patience=5	Validation Loss üst üste 5 epoch boyunca iyileşmezse eğitimi durdurur.
restore_best_weights=True	Eğitimi durdurduktan sonra, en iyi (en düşük val_loss) olan ağırlıkları geri yükler. Böylece model overfitting yapmadan en iyi halini otomatik alır. 
# Kaggle'dan aldığım URL'leri SentenceTransformer modeli (all-MiniLM-L6-v2) ile embedledim. Her URL ➔ 384 boyutlu bir vektör oldu.
# 
# BERT embedding'lerini eğitim ve test setine ayırdım.
# 
# 
# - Giriş: 384 boyutlu BERT embedding
# - Dense katmanlar: 256 ➔ 128 ➔ 64 nöron
# - Çıkış: 1 nöron (sigmoid aktivasyon, binary classification)
# 

# In[49]:


max_train_acc = max(history.history['accuracy'])
print(f" Eğitim (Train) Setinde En Yüksek Doğruluk: {max_train_acc:.4f}")
max_val_acc = max(history.history['val_accuracy'])
print(f" Doğrulama (Validation) Setinde En Yüksek Doğruluk: {max_val_acc:.4f}")


# In[50]:


# Loss Grafiği
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Accuracy Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# In[64]:


model.save("bert_ann_model.h5")


# In[58]:


X_embeddings_lstm = np.expand_dims(X_embeddings, axis=-1)


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings_lstm, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# In[60]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

model_lstm = Sequential([
    LSTM(128, input_shape=(384,1), return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_lstm.summary()


# LSTM(128)	BERT embedding üzerinden 128 adet gizli durum (hidden state) çıkarıyor
# 
# Dropout(0.3)	%30 dropout ➔ overfitting önlemek için
# 
# Dense(64)	64 nöronlu fully connected katman
# 
# Dropout(0.3)	Tekrar dropout ➔ daha güçlü regularization
# 
# Dense(1)	1 nöron (sigmoid aktivasyon) ➔ binary sınıflandırma (good vs bad URL)

# In[61]:


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history_lstm = model_lstm.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)


# Batch Size = 64 ➔ her adımda 64 örnek işleniyor.

# In[62]:


print(f"📈 En yüksek validation doğruluğu: {max(history_lstm.history['val_accuracy']):.4f}")


# In[63]:


import matplotlib.pyplot as plt

# Grafik boyutu
plt.figure(figsize=(14, 6))

# 1. Loss Grafiği
plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['loss'], label='Train Loss')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
plt.title('Loss Grafiği (BERT + LSTM)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# 2. Accuracy Grafiği
plt.subplot(1, 2, 2)
plt.plot(history_lstm.history['accuracy'], label='Train Accuracy')
plt.plot(history_lstm.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Grafiği (BERT + LSTM)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Tüm grafikleri hizala
plt.tight_layout()

# Göster
plt.show()


# In[65]:


model_lstm.save('bert_lstm_model.h5')


# In[ ]:




