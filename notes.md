Szervezéstől megkérdezni:
- lesz-e több autó az úton
- fix-e valami? autó, pálya?


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


kísérletezés:
    activation function: linear vagy sigmoid a hiddenbe?
    kell hidden?
    reward policy
    input
        szorzás, osztás átgondolás is

