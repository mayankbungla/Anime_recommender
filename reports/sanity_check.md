# Sanity check: CF vs content neighbours

Neighbours for five well-known anime, from each model.

## Death Note (id=1535)

**CF (learned SVD factors)**

```
  0.643  Shingeki no Kyojin
  0.594  Kiseijuu: Sei no Kakuritsu
  0.581  Fullmetal Alchemist: Brotherhood
  0.577  Code Geass: Hangyaku no Lelouch
  0.542  Code Geass: Hangyaku no Lelouch R2
  0.478  Seifuku Shojo The Animation
  0.478  Ningyou Tsukai
  0.456  Berserk: Ougon Jidai-hen III - Kourin
```

**Content (synopsis embeddings)**

```
  0.487  Kyougoku Natsuhiko: Kousetsu Hyaku Monogatari
  0.487  ChäoS;Child
  0.482  DRAMAtical Murder
  0.478  Kami nomi zo Shiru Sekai
  0.477  Death Parade
  0.471  Kuroshitsuji: Book of Murder
  0.465  Yami no Shihosha Judge
  0.464  Sen to Chihiro no Kamikakushi
```

## Steins;Gate (id=9253)

**CF (learned SVD factors)**

```
  0.672  Kimi no Na wa.
  0.627  Hunter x Hunter (2011)
  0.614  Fate/Zero 2nd Season
  0.612  Fullmetal Alchemist: Brotherhood
  0.609  Psycho-Pass
  0.608  Steins;Gate: Oukoubakko no Poriomania
  0.583  Fate/Zero
  0.566  Katanagatari
```

**Content (synopsis embeddings)**

```
  0.391  Tetsuwan Atom: Ivan no Wakusei - Robot to Ningen no Yuujou
  0.390  Kuromukuro
  0.382  Inferious Wakusei Senshi Gaiden Condition Green
  0.379  Yoru no Okite
  0.360  One Punch Man 2
  0.357  Little Busters!
  0.353  Kyuumei Senshi Nanosaver
  0.346  Planetarian: Chiisana Hoshi no Yume
```

## Cowboy Bebop (id=1)

**CF (learned SVD factors)**

```
  0.756  Cowboy Bebop: Tengoku no Tobira
  0.605  Mononoke Hime
  0.594  Black Jack
  0.574  Samurai Champloo
  0.568  Ghost in the Shell
  0.552  Shouwa Genroku Rakugo Shinjuu: Yotarou Hourou-hen
  0.547  Chuumon no Ooi Ryouriten (1991)
  0.540  Trigun
```

**Content (synopsis embeddings)**

```
  0.487  Happening Star ☆
  0.473  Dirty Pair: The Movie
  0.469  Black Bullet
  0.453  Hyper Police
  0.451  Ghost in the Shell 2: Innocence
  0.449  Malice@Doll
  0.444  DRAMAtical Murder
  0.444  Dirty Pair
```

## One Punch Man (id=30276)

**CF (learned SVD factors)**

```
  0.673  Boku no Hero Academia
  0.666  Mob Psycho 100
  0.613  Shokugeki no Souma
  0.550  Shokugeki no Souma: Ni no Sara
  0.533  Shingeki no Kyojin
  0.529  Hunter x Hunter (2011)
  0.516  Mobile Suit Gundam: Iron-Blooded Orphans
  0.511  Kiseijuu: Sei no Kakuritsu
```

**Content (synopsis embeddings)**

```
  0.515  One Punch Man 2
  0.504  Naruto: Shippuuden
  0.495  Hajime no Ippo: Champion Road
  0.473  Shijou Saikyou no Deshi Kenichi
  0.448  Boku no Hero Academia
  0.443  Evangelion: 1.0 You Are (Not) Alone
  0.439  Naruto
  0.435  Edokko Boy: Gatten Tasuke
```

## Shingeki no Kyojin (id=16498)

**CF (learned SVD factors)**

```
  0.643  Death Note
  0.533  One Punch Man
  0.522  Shingeki no Kyojin: Kuinaki Sentaku
  0.486  Code Geass: Hangyaku no Lelouch R2
  0.484  Fullmetal Alchemist: Brotherhood
  0.480  Kuroshitsuji: Book of Murder
  0.473  Kiseijuu: Sei no Kakuritsu
  0.468  Bleach
```

**Content (synopsis embeddings)**

```
  0.664  Shingeki no Kyojin Season 2
  0.430  No Game No Life Movie
  0.413  Pupa
  0.405  Evangelion: 1.0 You Are (Not) Alone
  0.398  Uchuu Senkan Yamato (Movie)
  0.393  Suisei no Gargantia
  0.388  Kinnikuman II Sei
  0.385  Saint Seiya: Legend of Sanctuary
```

## What I notice

_Write your observations here after reading the lists above:_
- Do the CF neighbours share a fanbase rather than just a genre?
- Do the content neighbours share plot/tone rather than just tags?
- Where do the two disagree, and which looks more sensible?
