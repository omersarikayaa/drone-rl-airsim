# STEP 6 - Kisa PPO entegrasyon egitimi

Bu adim ilk Stable-Baselines3 PPO entegrasyon testidir. Final performans beklenmez; amac PPO'nun `AirSimChaseEnv` ile `learn()` calistirabildigini, modeli kaydedebildigini ve kaydedilen modelin AirSim icinde tekrar yuklenebildigini dogrulamaktir.

Bu adimda YOLO, kamera, bbox, Target PPO, multi-agent RL, SAC/DQN veya uzun training yoktur.

## Paket kontrolu

```bash
python3 -c "import stable_baselines3, gymnasium, torch; print('ok')"
```

Eksikse:

```bash
python3 -m pip install stable-baselines3 gymnasium
```

Scriptler otomatik paket kurmaz.

## Kisa egitim

AirSimNH acik olmalidir.

Ilk test icin:

```bash
cd ~/drone_proje
python3 train_ppo_step6.py --timesteps 1000
```

Daha uzun dogrulama icin:

```bash
python3 train_ppo_step6.py --timesteps 5000
```

AirSim gercek zamanli calistigi icin egitim yavas olabilir. Bu normaldir.

Basarili egitimden sonra model burada olusur:

```text
models/ppo_chaser_step6.zip
```

Checkpointler:

```text
models/checkpoints/
```

Loglar:

```text
logs/
```

Beklenen final:

```text
STEP 6 PASSED: PPO training ran and model was saved.
```

## Evaluation

Training tamamlandiktan sonra:

```bash
python3 evaluate_ppo_step6.py --model models/ppo_chaser_step6.zip --steps 50
```

Beklenen log:

```text
[EVAL STEP 001] action=... safe=... reward=... distance=... front=... overridden=False collision=False
STEP 6 EVAL PASSED: trained PPO model loaded and ran in AirSim.
```

Model dosyasi yoksa once training calistirin:

```bash
python3 train_ppo_step6.py --timesteps 1000
```

## Notlar

Bu adimda modelin iyi kovalamasi beklenmez. Amac sadece entegrasyon dogrulamadir:

- Stable-Baselines3 PPO import ediliyor mu?
- `AirSimChaseEnv` Gymnasium uyumlu mu?
- `model.learn()` exception olmadan calisiyor mu?
- Model zip olarak kaydediliyor mu?
- Kaydedilen model tekrar yuklenip AirSim'de action uretiyor mu?

Ctrl+C ile durdurulursa script cleanup icin `env.close()` calistirmaya calisir.
