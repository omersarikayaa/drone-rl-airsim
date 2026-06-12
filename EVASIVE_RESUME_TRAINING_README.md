# Evasive Target ile Resume Training

Bu adim, mevcut iyi PPO modelini bozmadan daha dengeli `evasive` Target davranisi ile kisa ek egitim denemesi icindir.

Onemli notlar:

- Observation boyutu ayni kalir: `14`
- Action space ayni kalir: `Discrete(6)`
- Reward mantigi degismez
- Mevcut modeller silinmez
- `target_mode=simple` varsayilan olarak eski davranisi korur
- `target_mode=evasive` icin varsayilan hizlar daha sakindir:
  - `target_base_speed=1.2`
  - `target_escape_speed=1.5`

## Ortami Ac

```bash
cd ~/drone_proje
source .venv/bin/activate
```

## Iyi Modelden Evasive Resume Training

Uzun egitim baslatmadan once AirSimNH ve iki drone ayarlarinin hazir oldugundan emin ol.

```bash
python train_ppo_step6.py \
  --timesteps 5000 \
  --model-name ppo_chaser_evasive_plus5000 \
  --resume-from models/ppo_chaser_good_7000.zip \
  --target-mode evasive \
  --target-base-speed 1.2 \
  --target-escape-speed 1.5
```

## Yeni Modeli Evasive Target ile Calistir

```bash
python run_trained_ppo_agent.py \
  --model models/ppo_chaser_evasive_plus5000.zip \
  --steps 150 \
  --target-mode evasive \
  --target-base-speed 1.2 \
  --target-escape-speed 1.5
```

## Kisa Testler

Mevcut iyi modelin evasive Target ile calisip calismadigini hizli gormek icin:

```bash
python run_trained_ppo_agent.py --model models/ppo_chaser_good_7000.zip --steps 80 --target-mode evasive --target-base-speed 1.2 --target-escape-speed 1.5
```

Resume akisini uzun egitim yapmadan test etmek icin:

```bash
python train_ppo_step6.py --timesteps 256 --model-name ppo_chaser_evasive_resume_test --resume-from models/ppo_chaser_good_7000.zip --target-mode evasive --target-base-speed 1.2 --target-escape-speed 1.5
```

## Hedef Hizi Ayarlari

`target_base_speed`, Target uzak veya daha rahat durumdayken kullandigi temel hizdir.

`target_escape_speed`, Target yakin tehdit algiladiginda kullanabilecegi maksimum kacis hizidir. `TargetController`, final `vx/vy/vz` vektorunu bu maksimum hizla sinirlar.

Daha kolay yakalanabilir Target icin:

```bash
--target-base-speed 1.0 --target-escape-speed 1.3
```

Daha zor Target icin:

```bash
--target-base-speed 1.4 --target-escape-speed 1.8
```

Ilk resume denemelerinde `1.2 / 1.5` degerleri onerilir.
