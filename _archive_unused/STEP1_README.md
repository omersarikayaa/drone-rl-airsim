# STEP 1 - AirSim iki drone baglanti testi

Bu adim sadece AirSim icinde iki drone baglantisini, API control/arm/takeoff akisini, koordinat okumayi ve Chaser-Target arasi mesafe hesaplamayi test eder.

Bu adimda RL, PPO, YOLO, kamera, LiDAR, obstacle avoidance, reward veya egitim kodu yoktur.

## On kosullar

AirSimNH once acilmali ve sim calisir durumda olmalidir.

AirSim settings dosyasi genelde buradadir:

```bash
~/Documents/AirSim/settings.json
```

Gerekirse `step1_settings_example.json` icerigini mevcut `settings.json` dosyaniza kopyalayabilirsiniz. Bu proje mevcut `settings.json` dosyanizi otomatik olarak degistirmez veya ezmez.

Settings dosyasini dogrulamak icin:

```bash
python3 -m json.tool ~/Documents/AirSim/settings.json
```

## Testi calistirma

Bu ortamda proje klasoru:

```bash
cd ~/drone_proje
python3 step1_check_two_drones.py
```

Proje klasorunuz `~/drone_projem` ise ayni komutu o klasorde calistirin:

```bash
cd ~/drone_projem
python3 step1_check_two_drones.py
```

## Basarili cikti

Basarili calismada terminalde buna benzer loglar gorunmelidir:

```text
[OK] AirSim baglantisi kuruldu.
[INFO] AirSim vehicles: ['Chaser', 'Target']
[OK] API control enabled: Chaser
[OK] API control enabled: Target
[OK] Armed: Chaser
[OK] Armed: Target
[OK] Takeoff completed: Chaser
[OK] Takeoff completed: Target
[POS] Chaser: x=..., y=..., z=...
[POS] Target: x=..., y=..., z=...
[REL] Target relative to Chaser: dx=..., dy=..., dz=..., distance=...
[OK] Landing completed.
[OK] Cleanup completed.
STEP 1 PASSED: AirSim two-drone connection and coordinate reading works.
```

Bu cikti su sorulara net cevap verir:

1. AirSim baglantisi calisiyor mu?
2. Chaser gorunuyor mu?
3. Target gorunuyor mu?
4. Iki drone kalkiyor mu?
5. Koordinatlar okunuyor mu?
6. Aradaki mesafe hesaplaniyor mu?

## Sık karsilasilan hatalar

ECONNREFUSED hatasi AirSim'in acik olmadigini veya API portuna ulasilamadigini gosterir. AirSimNH'yi acip sim basladiktan sonra testi tekrar calistirin.

Chaser veya Target bulunamazsa `~/Documents/AirSim/settings.json` icindeki vehicle isimleri yanlis olabilir ya da AirSim settings degisikliginden sonra yeniden baslatilmamis olabilir.

Drone kalkmiyorsa terminaldeki API control, arm ve takeoff loglarina bakin. Hangi adimda hata verdigi loglarda acikca gorunmelidir.
