# Target Baslangic Konumunu Komuttan Ayarlama

Bu ozellik, `settings.json` dosyasini degistirmeden Target baslangic konumunu komuttan ayarlamayi saglar.

`--target-start-x` ve `--target-start-y`, reset sonrasi Chaser'a gore istenen `REL_GLOBAL dx/dy` ofsetidir. Chaser baslangic noktasinda duruyorsa bu degerler settings icindeki Target `X/Y` degeri gibi pratik calisir.

Kod reset sirasinda once global/world pose yontemlerini dener. AirSim `moveToPositionAsync` bazi kurulumlarda arac local NED frame ile calisirsa bunu loglar ve local-frame telafisiyle tekrar dener.

Observation shape degismez: `14`

Action space degismez: `Discrete(6)`

Reward sistemi degismez.

## Ortami Ac

```bash
cd ~/drone_proje
source .venv/bin/activate
```

## 15 Metre

```bash
python run_trained_ppo_agent.py \
  --model models/ppo_chaser_evasive_good_8000.zip \
  --steps 250 \
  --target-mode evasive \
  --target-base-speed 1.2 \
  --target-escape-speed 1.8 \
  --target-start-x 15 \
  --target-start-y 0
```

## 30 Metre

```bash
python run_trained_ppo_agent.py \
  --model models/ppo_chaser_evasive_good_8000.zip \
  --steps 350 \
  --target-mode evasive \
  --target-base-speed 1.2 \
  --target-escape-speed 1.8 \
  --target-start-x 30 \
  --target-start-y 0
```

## Capraz Baslangic

```bash
python run_trained_ppo_agent.py \
  --model models/ppo_chaser_evasive_good_8000.zip \
  --steps 400 \
  --target-mode evasive \
  --target-base-speed 1.2 \
  --target-escape-speed 1.8 \
  --target-start-x 30 \
  --target-start-y 15
```

## Uzak Baslangic ile Resume Training

```bash
python train_ppo_step6.py \
  --timesteps 3000 \
  --model-name ppo_chaser_far_30m_plus3000 \
  --resume-from models/ppo_chaser_evasive_good_8000.zip \
  --target-mode evasive \
  --target-base-speed 1.2 \
  --target-escape-speed 1.8 \
  --target-start-x 30 \
  --target-start-y 0
```

## Kisa Kontrol Testi

AirSimNH acikken:

```bash
python test_target_start_position.py
```

Beklenen log ornegi:

```text
[TARGET START TEST] requested x=15.00 y=0.00
[TARGET START TEST] actual dx=15.00 dy=0.00 distance=15.00
[TARGET START TEST] requested x=30.00 y=15.00
[TARGET START TEST] actual dx=30.00 dy=15.00 distance=33.54
TARGET START POSITION TEST PASSED
```
