# STEP 5 - Reward breakdown ve safety filter

Bu adim PPO egitimi degildir. Stable-Baselines3 import etmez, model egitmez, model kaydetmez ve Step 6/egitim baslatmaz.

Amac iki seyi dogrulamaktir:

- Reward sistemini okunabilir parcalara ayirmak.
- LiDAR sektor mesafeleri ve yukseklik sinirlari ile unsafe action secimlerini safety filter uzerinden duzeltmek.

## Safety filter

Agent veya PPO aday action secer. Safety filter bu action'i LiDAR sektorleri ve Chaser yuksekligi ile kontrol eder.

Risk varsa `safe_action` uretir ve drone `safe_action` uygular. Orijinal action kaybolmaz; `info` icinde hem orijinal hem safe action loglanir.

Action listesi:

```text
0 = FORWARD_TO_TARGET
1 = MOVE_LEFT
2 = MOVE_RIGHT
3 = MOVE_UP
4 = MOVE_DOWN
5 = HOVER
```

Ornek:

```text
original=0:FORWARD_TO_TARGET safe=2:MOVE_RIGHT overridden=True risk=danger
```

## Reward breakdown

Reward artik `reward_utils.py` icindeki `compute_chase_reward()` ile parcalara ayrilir:

```text
distance_delta_reward
catch_reward
collision_penalty
too_far_penalty
obstacle_penalty
safety_override_penalty
step_penalty
```

Toplam reward `reward_breakdown["total"]` olarak hesaplanir ve env step return icindeki `reward` degeriyle aynidir.

## Calistirma

AirSim integration kismi icin AirSimNH acik olmalidir. Pure unit-like testler AirSim olmadan da calisabilir.

```bash
cd ~/drone_proje
python3 test_step5_reward_safety_filter.py
```

## Beklenen cikti

Terminalde once pure testler gorunur:

```text
[SAFETY_UNIT] case=front_danger original=FORWARD_TO_TARGET safe=MOVE_RIGHT overridden=True reason=...
[REWARD_UNIT] case=distance_improved total=... breakdown=...
```

AirSim aciksa integration loglari gelir:

```text
[RESET] obs_shape=(14,)
[RESET_LIDAR] front=...
[STEP 001] original=0:FORWARD_TO_TARGET safe=0:FORWARD_TO_TARGET overridden=False risk=none reward=... distance=... front=... left=... right=... reason="" reward_parts={...}
STEP 5 PASSED: reward breakdown and safety filter integration works.
```

AirSim kismi basarisiz olur ama pure safety/reward testleri gecer ise:

```text
STEP 5 PARTIAL: pure safety/reward tests passed but AirSim integration failed.
```

## Notlar

Bu adimda safety filter sadece discrete action override yapar. Hiz azaltma, gelismis obstacle avoidance, kamera, YOLO, LiDAR path planning ve PPO training yoktur.

`safety_filter.py` ve `reward_utils.py` saf Python dosyalaridir; AirSim import etmezler.
