Szervezéstől megkérdezni:
- lesz-e több autó az úton
- fix-e valami? autó, pálya?
VÁLASZ: pénteken megtudjuk


tasks:
- feltenni gui nélküli linux ot
- működésre bírni a docker környezetet
- párhuzamosan futtatni a gui nélküli linuxokban
    a docker-es környezetet
- inference módban is logolni a fitness-t meg ilyeneket
    mert jelenleg az csak train során megy.
    informatív lehet.
    szóval mode-tól függetleníteni a futásközbeni
    a console ra logolást
- ellenőrizni, hogy lap teljesítéskor kap-e 
    nagy negatív pontot valahogy a distraced-ből
- logolás javítás, nem csak a megjelenítés,
    hanem a logika a kódban, mert most all over the place
    printf ek vannak itt ott keresztül kasul
- megnézni a warning okat visual studio ban amikor elindul
    a projekt
- a hálókat úgy menteni, hogy a legjobb legelol legyen?
- ellenőrizni, hogy Crestart vagy Cinit során kap-e pontokat a háló
    szóval van-e olyan állapot, amikor az eveluate fut, pedig még
    nem kéne, és közbe a cs.distRaced is faszságot ad vissza
- a logolásban legyen benne, hogy épp mivel próbálkozott mint:
    hidden layer van-e, neuronok száma, milyen multiplier...
    mert nem fogjuk tudni, hogy épp mivel lett már elindítva
    ha arrébb tesszük a logfile-t


kísérletezés:
    activation function: linear vagy sigmoid a hiddenbe?
    kell hidden?
    reward policy
    input
        szorzás, osztás átgondolás is

