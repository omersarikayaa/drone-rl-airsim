# STEP 3 - PPO-style AirSim environment iskeleti

Bu adim PPO training degildir. Stable-Baselines3 ile egitim baslatmaz, model kaydetmez ve model yuklemez.

Bu adim sadece AirSim icin Gymnasium-style bir environment iskeleti kurar ve `reset()` / `step(action)` arayuzunu random actionlarla dogrular.

## Mantik

`AirSimChaseEnv` icinde:

- Chaser action alan ajan olarak dusunulur.
- Target simdilik scriptli hareket eder.
- Global/world pozisyonlar `simGetObjectPose` ile okunur.
- Observation 8 boyutlu sayisal vektordur.
- Reward basit mesafe degisimi, catch, collision ve too-far kontrolu ile hesaplanir.
- `terminated` ve `truncated` degerleri uretilir.

Observation vektoru:

```text
[
  dx_normalized,
  dy_normalized,
  dz_normalized,
  distance_normalized,
  chaser_x_normalized,
  chaser_y_normalized,
  target_x_normalized,
  target_y_normalized
]
```

Gymnasium yüklu degilse environment yine calisir. Gymnasium varsa `action_space` ve `observation_space` da tanimlanir.

## Calistirma

AirSimNH once acik olmalidir. `~/Documents/AirSim/settings.json` icinde `Chaser` ve `Target` bulunmalidir.

```bash
cd ~/drone_proje
python3 test_step3_env_random_actions.py
```

## Beklenen cikti

Terminalde buna benzer loglar gorunmelidir:

```text
[STEP3] PPO-style AirSimChaseEnv random action test
[INFO] This script does not train PPO. It only validates reset/step.
[RESET] obs_shape=(8,) obs=[...]
[STEP 001] action=0:FORWARD_TO_TARGET reward=... distance=... dx=... dy=... collision=False terminated=False truncated=False
[STEP 002] action=5:HOVER reward=... distance=... dx=... dy=... collision=False terminated=False truncated=False
STEP 3 PASSED: PPO-style AirSimChaseEnv reset/step interface works.
```

## Gozlem

AirSim ekraninda Chaser ve Target kalkmalidir.

Target scriptli hareket etmelidir.

Chaser random actionlara gore hareket etmelidir.

Her step logunda distance, reward, collision, terminated ve truncated degerleri gorunmelidir.

Test sonunda iki drone guvenli sekilde inmeli, disarm olmali ve API control kapanmalidir.

## Hata durumlari

AirSim baglanti hatasi alirsaniz AirSimNH'nin acik ve sim'in baslamis oldugunu kontrol edin.

`Chaser` veya `Target` bulunamazsa `~/Documents/AirSim/settings.json` icindeki vehicle isimlerini kontrol edin ve AirSimNH'yi yeniden baslatin.

`numpy` bulunamazsa once numpy kurulmalidir. Bu environment sayisal observation icin numpy gerektirir.

Bu adimda PPO import edilmez, Stable-Baselines3 kullanilmaz ve egitim yapilmaz.
