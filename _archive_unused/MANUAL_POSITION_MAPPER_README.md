# Manual Position Mapper

Bu script, AirSim icinde sadece secilen tek drone'u terminal komutlariyla hareket ettirip canli `GLOBAL/WORLD` koordinatini gormek ve noktalari JSON dosyasina kaydetmek icindir.

Diger drone'a `enableApiControl`, `arm`, `takeoff` veya hareket komutu gonderilmez.

Koordinat okuma icin `simGetObjectPose(vehicle_name)` kullanilir.

## Calistirma

```bash
cd ~/drone_proje
source .venv/bin/activate
python manual_position_mapper.py --vehicle Chaser --speed 2.0 --z -5.0 --output mapped_positions.json
```

Target icin:

```bash
python manual_position_mapper.py --vehicle Target --speed 2.0 --z -5.0 --output mapped_positions.json
```

## Komutlar

Komutlar `command>` satirina yazilip Enter ile calistirilir.

```text
w 2      -> +X yonunde 2 saniye git
s 2      -> -X yonunde 2 saniye git
a 2      -> -Y yonunde 2 saniye git
d 2      -> +Y yonunde 2 saniye git
q 1      -> yukari cik, z daha negatif
e 1      -> asagi in, z daha pozitif
h        -> hover
pos      -> global pozisyonu yazdir
p isim   -> mevcut global pozisyonu kaydet
land     -> inis yap, disarm, API kapat ve cik
exit     -> hover yap, API kapat ve cik
```

Her hareketten sonra pozisyon yazdirilir:

```text
[POS] vehicle=Chaser x=14.82 y=6.35 z=-5.02
```

## Nokta Kaydetme

```text
command> p tree_front
```

`mapped_positions.json` varsa mevcut kayitlar okunur ve yeni nokta ustune eklenir.

Kayit ornegi:

```json
{
  "tree_front": {
    "vehicle": "Chaser",
    "x": 14.82,
    "y": 6.35,
    "z": -5.02
  }
}
```

## Cikis

Guvenli inis:

```text
command> land
```

Inmeden cikis:

```text
command> exit
```

`Ctrl+C` de `exit` gibi hover ve API kapatma cleanup'ini calistirir.
