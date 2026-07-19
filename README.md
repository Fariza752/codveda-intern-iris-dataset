# Iris Növ Təsnifatı — KMeans Clustering

## 📌 Data haqqında

**Mənbə:** [Iris Species Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/iris) *(linkə klik et, birbaşa Kaggle səhifəsi açılır)*

Klassik Iris dataset-i — 150 sətir, 3 çiçək növü (`setosa`, `versicolor`, `virginica`), hər növdən 50 nümunə. 4 numerik ölçü var: `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (sm ilə).

## 🎯 Niyə bu analiz aparılır?

Bu layihədə **unsupervised learning (nəzarətsiz öyrənmə)** — yəni növ etiketlərini (label) istifadə etmədən, yalnız ölçülərə əsaslanaraq, çiçəkləri özbaşına qruplara (cluster) ayırmaq məqsəd qoyulub.

**Data-dan nə istəyirik:**
1. Ölçülərə görə çiçəkləri neçə təbii qrupa bölmək olar — insan bilmədən model tapa bilərmi?
2. KMeans-in tapdığı cluster-lər real növlərlə (setosa/versicolor/virginica) nə qədər üst-üstə düşür?
3. Hansı ölçü (petal, yoxsa sepal) növləri daha aydın ayırır?

---

## 🔍 Kəşfiyyat Analizi (EDA)

**Sepal Length paylanması (histogram + KDE)** — Nə göstərir: gülyarpağın uzunluğunun tezlik paylanması.
**Nəticə:** Təxminən normal paylanmaya yaxın, 5-6.5 sm arası ən çox rast gəlinir.

**Sepal Width paylanması (histogram + KDE)** — Nə göstərir: gülyarpağın enliliyinin paylanması.
**Nəticə:** 3 sm ətrafında zirvə (peak) var, simmetrik paylanma.

**Petal Length paylanması (histogram + KDE)** — Nə göstərir: ləçəyin uzunluğunun paylanması.
**Nəticə:** İki aydın qrup görünür (1-2 sm və 3-7 sm) — bu, artıq setosa-nın digər növlərdən fərqləndiyinə işarədir.

**Petal Width paylanması (histogram + KDE)** — Nə göstərir: ləçəyin enliliyinin paylanması.
**Nəticə:** Petal length kimi, burada da iki qrup aydın seçilir — petal ölçüləri növ ayırmaqda daha güclü siqnaldır.

**Növlərin sayı (countplot)** — Nə göstərir: hər növdən neçə nümunə olduğu.
**Nəticə:** Data tam balanslıdır — hər növdən düz 50 nümunə (150 sətir).

**Outlier-lər — Petal Width, Petal Length, Sepal Width, Sepal Length (boxplot-lar)** — Nə göstərir: hər ölçüdə kənar dəyərlərin olub-olmaması.
**Nəticə:** Yalnız `sepal_width`-də bir neçə kənar (outlier) dəyər aşkarlandı, digər 3 ölçüdə əhəmiyyətli outlier yox idi.

**Korrelyasiya matrisi (heatmap)** — Nə göstərir: 4 ölçü arasında xətti əlaqə gücü.
**Nəticə:** `petal_length` və `petal_width` arasında çox güclü müsbət korrelyasiya var (~0.96) — bu iki ölçü demək olar birlikdə dəyişir və növ ayırmaqda ən informativ cütdür.

**Pairplot (bütün ölçülərin cüt-cüt müqayisəsi, növə görə rənglənmiş)** — Nə göstərir: hər ölçü cütünün scatter-i, 3 növün rənglə fərqləndirilməsi.
**Nəticə:** Petal ölçüləri əsasında setosa digər iki növdən tam aydın ayrılır, versicolor və virginica arasında isə qismən üst-üstə düşmə var.

---

## 🧹 Data Cleaning (Təmizləmə)

- `df.isnull().sum()` — **heç bir boş (null) dəyər tapılmadı**, data tam təmiz idi
- `df.shape` → 150 sətir, 5 sütun (4 numerik + 1 kateqorik `species`)
- **Outlier-lərin təmizlənməsi (IQR metodu):** yalnız `sepal_width` sütununda outlier aşkarlandığı üçün, Q1-1.5×IQR / Q3+1.5×IQR sərhədlərindən kənar dəyərlər clip edildi (sərhədə çəkildi, silinmədi)

**Sepal Width — clip-dən sonra (boxplot)** — Nə göstərir: təmizlənmiş sütunun yeni vəziyyəti.
**Nəticə:** Kənar dəyərlər sərhədə çəkildi, paylanma daha stabil oldu.

---

## ⚙️ Feature Engineering (Encoding və hazırlıq)

- Model üçün yalnız 4 numerik ölçü (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) seçildi — `species` sütunu **istifadə olunmadı**, çünki bu unsupervised layihədir (növ adı sonradan yalnız yoxlama üçün istifadə olunur)
- **StandardScaler** tətbiq edildi — KMeans məsafə əsaslı alqoritm olduğu üçün bütün ölçülər eyni miqyasa gətirilməlidir (əks halda böyük ədədli sütun süni şəkildə üstünlük edərdi)

---

## 🤖 Model və nəticələr

### Optimal cluster sayının tapılması

**Elbow metodu (inertia vs k)** — Nə göstərir: cluster sayı artdıqca daxili variasiyanın (inertia) necə azaldığı.
**Nəticə:** Əyri k=3-də aydın "dirsək" (elbow) yaradır — optimal cluster sayı 3-dür.

**Silhouette Score (k=2-10)** — Nə göstərir: hər k dəyəri üçün cluster-lərin nə qədər yaxşı ayrıldığını ölçən bal.
**Nəticə:** Elbow metodunu təsdiqləyir — k=3 ən yüksək silhouette bal verir.

### Final model: KMeans (k=3)

Cluster ölçüləri: Cluster 0 — 53 nümunə, Cluster 1 — 50 nümunə, Cluster 2 — 47 nümunə.

**Cluster paylanması (pie chart)** — Nə göstərir: hər cluster-in ümumi datadakı faiz payı.
**Nəticə:** Cluster-lər təxminən bərabər ölçülüdür (35.3% / 33.3% / 31.3%), real növ balansına çox yaxındır.

**KMeans Cluster-lər + Centroid-lər (scatter plot, petal_length vs petal_width)** — Nə göstərir: 3 cluster-in petal ölçüləri üzrə vizual ayrılması və hər cluster-in mərkəzi.
**Nəticə:** Cluster 1 (kiçik petal ölçülü) tam aydın ayrılır, digər iki cluster arasında sərhəddə qismən üst-üstə düşmə var.

### Real növlərlə müqayisə (Cross-tab)

| Species \ Cluster | 0 | 1 | 2 |
|---|---|---|---|
| setosa | 0 | 50 | 0 |
| versicolor | 39 | 0 | 11 |
| virginica | 14 | 0 | 36 |

**Cross-tab Heatmap (True Species vs KMeans Cluster)** — Nə göstərir: hər real növün hansı cluster-lərə düşdüyünü rəng intensivliyi ilə göstərir.
**Nəticə:** `setosa` 100% dəqiqliklə Cluster 1-ə düşüb. `versicolor` və `virginica` arasında qarışıqlıq var — 11 versicolor Cluster 2-yə, 14 virginica isə Cluster 0-a düşüb.

**Ümumi model dəqiqliyi: 83.33%**

---

## 💡 Business / Elmi Insight-lar

1. **Petal ölçüləri növ ayırmaqda sepal ölçülərindən qat-qat güclüdür** — real tətbiqlərdə yalnız petal ölçmək kifayət edə bilər.
2. **Setosa növü digər ikisindən tam aydın ayrılır** — avtomatlaşdırılmış sortlaşdırma sistemlərində 100% etibarla tanına bilər.
3. **Versicolor və virginica arasındakı qarışıqlıq** göstərir ki, bu iki növ morfoloji cəhətdən yaxındır — tam dəqiqlik üçün əlavə ölçülər və ya supervised model lazımdır.
4. Label olmadan da (unsupervised), yalnız ölçülərə əsaslanaraq real qruplara çox yaxın nəticə əldə etmək mümkündür.

## 🛠️ Texnologiyalar
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (KMeans, StandardScaler, silhouette_score)

## 📁 Fayllar
- `Iris.ipynb` — bütün analiz, vizuallaşdırma və clustering kodları
