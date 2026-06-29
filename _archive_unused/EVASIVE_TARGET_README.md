# Evasive Target Controller

Bu asama Target drone'un daha akilli kacmasini saglar. Target PPO degildir; Target icin RL, multi-agent RL, YOLO veya kamera/bbox eklenmemistir.

Chaser mevcut PPO modelini kullanmaya devam eder. Observation shape 14, action space 6 ve reward yapisi uyumlu kalir.

## Mantik

`target_controller.py` icindeki `TargetController` kuralli bir kacis mantigi kullanir:

- Chaser yakindaysa Target Chaser'dan uzaklasan yonde kacar.
- Chaser cok yakindaysa yanal manevra ekler.
- Chaser uzaktaysa duz/hafif yanal hareket eder.
- Target LiDAR sektorleri varsa onde engel gorunce ileri gitmekten kacinir.

Varsayilan env davranisi hala eski simple target modudur:

```python
AirSimChaseEnv(target_mode="simple")
```

Yeni kacis modu:

```python
AirSimChaseEnv(target_mode="evasive")
```

## Test

```bash
cd ~/drone_proje
source .venv/bin/activate
python test_evasive_target_controller.py
```

Beklenen final:

```text
EVASIVE TARGET TEST PASSED: TargetController evasive behavior works.
```

## PPO demo

Egitilmis iyi Chaser modelini evasive Target'a karsi calistirma:

```bash
python run_trained_ppo_agent.py \
  --model models/ppo_chaser_good_7000.zip \
  --steps 150 \
  --target-mode evasive
```

Bu komut egitim yapmaz. Sadece modeli yukler ve AirSim icinde calistirir.

## Resume training hazirligi

Evasive Target testi calisiyorsa, sonraki asamada iyi baseline modelden devam egitim baslatilabilir:

```bash
python train_ppo_step6.py \
  --timesteps 5000 \
  --model-name ppo_chaser_evasive_plus5000 \
  --resume-from models/ppo_chaser_good_7000.zip \
  --target-mode evasive
```

Bu README'deki resume komutu hazirlik icindir. Bu adimda uzun PPO egitimi baslatilmaz.

## Notlar

Evasive Target daha zor bir senaryo uretir. Chaser ilk denemelerde Target'i yakalayamazsa bu normaldir.

Simple mode bozulmamistir; Step 3/4/5 testleri default olarak simple mode ile calismaya devam eder.
