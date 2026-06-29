# STEP 2 - Scriptli Target hareket testi

Bu adimin amaci Target drone'u basit ve kuralli bir rota ile hareket ettirmektir. Chaser bu adimda hedefi kovalamaz; sadece guvenli yukseklikte hover yapar ve referans olarak kullanilir.

Bu adimda PPO, RL, Stable-Baselines3, Gymnasium, YOLO, kamera, bbox, reward veya egitim kodu yoktur.

## On kosullar

AirSimNH once acilmali ve sim calisir durumda olmalidir.

`~/Documents/AirSim/settings.json` icinde iki arac bulunmalidir:

```text
Chaser
Target
```

Step 1'de dogrulandigi gibi global/world pozisyon okumasi icin `simGetObjectPose` kullanilir.

## Testi calistirma

```bash
cd ~/drone_proje
python3 step2_scripted_target_motion.py
```

## Beklenen cikti

Terminalde buna benzer loglar gorunmelidir:

```text
[STEP2] Scripted Target Motion Test
[INFO] This test does not train PPO. It only moves Target with scripted commands.
[OK] AirSim connected.
[INFO] Vehicles: ['Chaser', 'Target']
[OK] Takeoff completed.
[OK] Safe altitude reached: z=-5.0
[INFO] Chaser hovering.
[SEGMENT 1] Target moving to x=..., y=..., z=-5.00
[GLOBAL_POS] Chaser: x=..., y=..., z=...
[GLOBAL_POS] Target: x=..., y=..., z=...
[REL_GLOBAL] dx=..., dy=..., dz=..., distance=...
[OK] Scripted target motion completed.
[OK] Landing completed.
[OK] Cleanup completed.
STEP 2 PASSED: Target scripted motion works.
```

## Gozlem

AirSim ekraninda Chaser sabit hover yapmalidir.

Target segment segment hareket etmelidir:

1. X ekseninde +5 metre
2. Y ekseninde +5 metre
3. X ekseninde +5 metre daha
4. Y ekseninde -5 metre
5. Kisa hover

Her segmentten sonra `[REL_GLOBAL]` distance degeri yazdirilir ve hareket boyunca degismelidir.

Test sonunda iki drone da guvenli sekilde inmelidir.

## Hata durumlari

`Chaser` veya `Target` gorunmuyorsa `~/Documents/AirSim/settings.json` icindeki arac isimlerini kontrol edin ve AirSimNH'yi yeniden baslatin.

`STEP 2 FAILED: Target did not move in global coordinates.` cikarsa Target komut aliyor gibi gorunse bile `simGetObjectPose` pozisyonu degismiyor demektir.

`STEP 2 PARTIAL: takeoff works but target motion was not verified.` cikarsa kalkis basarili olmustur ama Target hareketi veya mesafe degisimi yeterince dogrulanamamistir.
