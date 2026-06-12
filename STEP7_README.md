# STEP 7 - Egitilmis PPO modelini demo modunda calistirma

Bu adim egitim degildir. Yeni PPO training baslatmaz ve `model.learn()` kullanmaz. Amac Step 6'da kaydedilen PPO modelini AirSim icinde temiz bir demo/run scripti ile calistirmaktir.

Varsayilan model:

```text
models/ppo_chaser_step6.zip
```

## Calistirma

AirSimNH acik olmalidir.

```bash
cd ~/drone_proje
source .venv/bin/activate
python run_trained_ppo_agent.py --model models/ppo_chaser_step6.zip --steps 100
```

Kisa bekleme eklemek isterseniz:

```bash
python run_trained_ppo_agent.py --model models/ppo_chaser_step6.zip --steps 100 --delay 0.1
```

## Beklenen davranis

Chaser ve Target kalkar.

Target scriptli hareket etmeye devam eder.

Chaser PPO modelinin sectigi actionlarla hedefi kovalar.

Distance azalirsa model iyi davranmaya baslamis demektir.

`caught=True` veya `reason=caught` gorunurse hedef yakalanmistir.

Test sonunda iki drone inmeli, disarm olmali ve API control kapanmalidir.

## Log alanlari

```text
action
```

PPO modelinin sectigi action.

```text
safe
```

Safety filter sonrasi drone'a uygulanan action.

```text
distance
```

Chaser-Target global mesafesi.

```text
front
```

Chaser LiDAR on sektor mesafesi.

```text
overridden
```

Safety filter action degistirdi mi?

```text
caught
```

Target yakalandi mi?

```text
reason
```

Episode bitis sebebi: `caught`, `collision`, `too_far`, `max_steps` veya `none`.

## Basarili cikti

Ornek:

```text
[STEP7] Run trained PPO Chaser agent
[INFO] This script does not train PPO. It only loads and runs a trained model.
[INFO] model=/home/omer/drone_proje/models/ppo_chaser_step6.zip
[RESET] obs_shape=(14,) distance=5.00 lidar_available=True lidar_front=50.00
[RUN STEP 001] action=0:FORWARD_TO_TARGET safe=0:FORWARD_TO_TARGET reward=0.91 distance=4.54 caught=False reason=none
[RUN STEP 009] action=0:FORWARD_TO_TARGET safe=0:FORWARD_TO_TARGET reward=100.56 distance=1.98 caught=True reason=caught
[RESULT] Episode ended at step 9.
[RESULT] reason=caught
[RESULT] final_distance=1.98
STEP 7 SUCCESS: Chaser caught Target.
STEP 7 PASSED: trained PPO agent ran in AirSim demo mode.
```

Model yoksa once Step 6 training calistirin:

```bash
python train_ppo_step6.py --timesteps 1000
```
