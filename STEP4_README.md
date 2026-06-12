# STEP 4 - Observation + LiDAR sektor mesafeleri

Bu adim PPO egitimi degildir. Stable-Baselines3 kullanmaz, model egitmez ve safety filter uygulamaz.

Amac `AirSimChaseEnv` observation vektorunu 8 boyuttan 14 boyuta cikarmaktir. PPO ileride hem hedef bilgisini hem de Chaser uzerindeki LiDAR engel mesafelerini gorebilecektir.

## Observation

Ilk 8 deger hedef ve pozisyon bilgisidir:

```text
dx_normalized
dy_normalized
dz_normalized
distance_normalized
chaser_x_normalized
chaser_y_normalized
target_x_normalized
target_y_normalized
```

Son 6 deger LiDAR sektor mesafeleridir:

```text
front_dist_normalized
front_left_dist_normalized
front_right_dist_normalized
left_dist_normalized
right_dist_normalized
back_dist_normalized
```

LiDAR mesafeleri `MAX_LIDAR_DISTANCE = 50.0` ile normalize edilir. Engel yoksa veya LiDAR verisi gelmezse sektorler fallback olarak 50 metre kabul edilir ve normalized degerler 1.0 civarina gelir.

Raw sektor mesafeleri `info` icinde su alanlarla gorulur:

```text
lidar_front
lidar_front_left
lidar_front_right
lidar_left
lidar_right
lidar_back
```

## Calistirma

AirSimNH acik olmalidir. `~/Documents/AirSim/settings.json` icinde Chaser ve Target bulunmali, Chaser uzerinde `Lidar1` enabled olmalidir.

```bash
cd ~/drone_proje
python3 test_step4_observation_lidar.py
```

## Beklenen cikti

Terminalde buna benzer loglar gorunmelidir:

```text
[STEP4] Observation + LiDAR validation test
[RESET] obs_shape=(14,)
[RESET] distance=...
[RESET] lidar_available=True
[RESET] lidar_point_count=...
[RESET_LIDAR] front=..., front_left=..., front_right=..., left=..., right=..., back=...
[STEP 001] action=0:FORWARD_TO_TARGET reward=... distance=... lidar_available=True points=... front=... left=... right=... back=...
STEP 4 PASSED: observation includes target-relative state and LiDAR sector distances.
```

LiDAR verisi gelmezse environment yine calisir ve `MAX_LIDAR_DISTANCE` fallback kullanir:

```text
STEP 4 PARTIAL: observation shape works but LiDAR data is not available.
```

Bu durumda `~/Documents/AirSim/settings.json` icinde `Lidar1` sensorunun Chaser icin enabled oldugunu kontrol edin.

## Gozlem

AirSim ekraninda Chaser ve Target kalkmalidir.

Target scriptli hareket etmelidir.

Chaser actionlara gore hareket etmelidir.

AirSim'de LiDAR debug noktalarini gorebilirsiniz.

Test sonunda iki drone guvenli sekilde inmeli, disarm olmali ve API control kapanmalidir.
