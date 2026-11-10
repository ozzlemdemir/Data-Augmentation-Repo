# Data-Augmentation-2026

Bu proje, tarımsal görüntülerden oluşan YOLO veri kümesini Johnsongrass sınıfına göre ayırır ve görüntü artırma işlemleri için hazırlar. Görseller `.jpg`/`.png`, nesne etiketleri ise YOLO biçiminde `.txt` dosyalarında tutulur.

## Klasörler

- `orijinal_veriler_2026/`: Kaynak görseller, etiketler ve sınıf adlarını içeren `classes.txt`.
- `Klasor_1(Johsongrassli)/`: Johnsongrass (sınıf kimliği `3`) içeren görsel-etiket çiftleri; `train`, `test` ve `val` bölümlerine ayrılır.
- `Klasor_2(Diger)/`: Sınıf `3` içermeyen çiftler; eğitim verisi yedi artırma tekniğine göre alt klasörlere bölünür.
- `Scriptler/`: Veri ayırma ve artırma Python betikleri.
- `augmented_data/`, `final_set/`: Sonuçların düzenlenmesi için ayrılmış klasörler.

## Betikler

- `Scriptler/find_johnsongrass.py`: Johnsongrass içeren çiftleri bulur; sabit rastgele tohumla yaklaşık `%80/%10/%10` oranında ayırır.
- `Scriptler/find_others.py`: Johnsongrass içermeyen çiftleri seçer ve veri bölümlerini oluşturur.
- `Scriptler/augmentations/aug_*.py`: Yatay çevirme, ölçekleme, döndürme, shear, HSV, gölge ve kırpma artırmalarını uygular; YOLO etiketlerini de günceller.
- `Scriptler/augmentations/run_klasor1.py` ve `run_klasor2.py`: Artırma betiklerini ilgili eğitim verisine uygular. İlk klasörde her tekniği eğitim görsellerine ayrı ayrı, ikinci klasörde ise her alt kümeye atanmış tekniği uygular.

Betikleri çalıştırmadan önce kaynak ve hedef klasör yollarının betiklerdeki `ROOT` ayarıyla eşleştiğinden emin olun. Artırma betikleri OpenCV (`cv2`) ve NumPy kullanır.
