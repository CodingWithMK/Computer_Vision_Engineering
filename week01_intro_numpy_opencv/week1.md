# Computer Vision – 1. Hafta Uygulama Föyü

Bu depo, mikro-öğrenme prensibine uygun olarak tasarlanmış **Computer Vision** (Bilgisayarla Görü) mühendisliği yolculuğunun **1. haftasına** ait detaylı uygulama föyünü içerir. Her gün için **20 dakika teori** + **10–15 dakika pratik kod** egzersizleri, Cumartesi mini proje ve Pazar tekrar & quiz bölümleri bulunur.

---

## 🚀 Genel Yapı

* **Hafta:** Pazartesi–Cuma
* **Günlük Seans:** 20 dk teori + 10–15 dk kod uygulaması
* **Cumartesi:** Mini proje
* **Pazar:** Tekrar & quiz
* **Araçlar:** Python 3.8+, virtualenv, Git/GitHub, NumPy, OpenCV, Matplotlib

### Kurulum

```bash
# 1. virtualenv oluştur
python3 -m venv cv_env
# 2. Ortamı aktive et
source cv_env/bin/activate    # Linux/Mac
# veya
cv_env\Scripts\activate     # Windows
# 3. Gerekli paketleri kur
pip install numpy opencv-python matplotlib
# 4. Bu repo'yu klonla
git clone https://github.com/kullanici/cv-workshop.git
cd cv-workshop
```

---

## 📅 Günlük Konular

### Gün 1: Ortam Kurulumu & Git/GitHub

**Anahtar Adımlar:**

1. Python ve virtualenv kurulumu
2. `pip install` ile kütüphane yükleme
3. Git konfigürasyonu (`git config`)
4. GitHub’da yeni repo oluşturma ve ilk push

<details>
<summary>Örnek Komutlar</summary>

```bash
# Git ayarları
git config --global user.name "Adınız Soyad"
git config --global user.email "email@ornek.com"
# Repo oluştur, init, commit, push
mkdir cv_workshop && cd cv_workshop
git init
echo "# CV Workshop" > README.md
git add .
git commit -m "İlk commit: ortam ve README"
git remote add origin https://github.com/kullanici/cv-workshop.git
git push -u origin main
```

</details>

### Gün 2: NumPy ile Matris İşlemleri

* **np.array**: `np.array(object, dtype=None, copy=True, order='K', ndmin=0)`
* **np.zeros / np.ones**: `np.zeros(shape, dtype=float)` / `np.ones(shape)`
* **np.eye**: `np.eye(N, M=None, k=0)`
* **reshape / T**: `a.reshape(newshape)`, `a.T`
* **np.dot / np.matmul**: Skaler ve matris çarpımı
* **np.random.rand**: Rastgele değerli dizi
* **np.linalg.det / inv**: Determinant ve matris tersi

<details>
<summary>Kod Örneği</summary>

```python
import numpy as np
# Array oluşturma
a = np.array([1,2,3], dtype=float)
Z = np.zeros((2,3), dtype=int)
I = np.eye(3)
A = np.arange(6).reshape((2,3))
print(a, Z, I, A)
# Matris işlemleri
B = np.random.rand(3,3)
detB = np.linalg.det(B)
invB = np.linalg.inv(B)
print("det(B)=", detB)
print("inv(B)=", invB)
```

</details>

### Gün 3: OpenCV – Görüntü Okuma & Gösterme

* **cv2.imread**: `cv2.imread(filename, flags)`
* **cv2.cvtColor**: `cv2.cvtColor(src, code)`
* **cv2.imshow / waitKey / destroyAllWindows**
* **cv2.imwrite**: `cv2.imwrite(filename, img, params=None)`

<details>
<summary>Kod Örneği</summary>

```python
import cv2
# Görüntü oku
gri = cv2.imread('resim.jpg', cv2.IMREAD_GRAYSCALE)
renk = cv2.imread('resim.jpg')
# Göster
cv2.imshow('Gri', gri)
if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('gri_kayit.png', gri)
cv2.destroyAllWindows()
```

</details>

### Gün 4: Matplotlib ile Görselleştirme

* **plt.figure**: `plt.figure(figsize, dpi)`
* **plt.plot**: `plt.plot(x, y, fmt, label)`
* **plt.imshow**: `plt.imshow(X, cmap, interpolation)`
* **plt.title / xlabel / ylabel**
* **plt.subplot**
* **plt.show**

<details>
<summary>Kod Örneği</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,2*np.pi,100)
y = np.sin(x)
plt.figure(figsize=(6,4))
plt.plot(x,y, linestyle='--', marker='o', label='sin(x)')
plt.title('Sinüs Grafiği')
plt.xlabel('Açı [rad]')
plt.ylabel('Genlik')
plt.legend()
plt.show()
```

</details>

### Gün 5: Lineer Cebir Özeti

* **np.dot / np.cross / np.linalg.norm**
* **np.linalg.eig / np.linalg.svd**

<details>
<summary>Kod Örneği</summary>

```python
import numpy as np
# Skaler ve dış çarpım
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
print(np.dot(v1,v2))
print(np.cross(v1,v2))
# Norm
print(np.linalg.norm(v1, ord=2))
# Özdeğer & SVD
A = np.array([[2,0],[0,3]])
w,v = np.linalg.eig(A)
print(w, v)
B = np.random.rand(2,3)
U,S,Vh = np.linalg.svd(B)
print(S)
```

</details>

---

## 🎯 Cumartesi Mini Proje

1. 3×3 rastgele matris oluşturun.
2. Determinant ve inverseni hesaplayın.
3. Özdeğer/özvektör ve SVD uygulayın.
4. Skaler & dış çarpım ile vektör normlarını kıyaslayın.

---

> **İpuçları:**
>
> * Her işlemi adım adım CLI veya Jupyter Notebook’ta deneyin.
> * Kod bloklarını anlamaya odaklanın, parametrelerin etkisini değiştirmeyi deneyin.
> * Sorularınızı Issues bölümünde paylaşabilirsiniz.

---

### Lisans

MIT © 2025
