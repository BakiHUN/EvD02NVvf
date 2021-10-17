#include "CDriver.h"
#include "genann.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <time.h>


/* Gear Changing Constants*/
const int gearUp[6] =
{
    5000,6000,6000,6500,7000,0
};
const int gearDown[6] =
{
    0,2500,3000,3000,3500,3500
};

/* Clutch constants */
const float clutchMax = 0.5;
const float clutchDelta = 0.05;
const float clutchRange = 0.82;
const float clutchDeltaTime = 0.02;
const float clutchDeltaRaced = 10;
const float clutchDec = 0.01;
const float clutchMaxModifier = 1.3;
const float clutchMaxTime = 1.5;

int stuck;
float clutch;


// CUSTOM STUFF FROM HERE
float prevDamage = 0.0f;
float prevDistRaced = 0.0f;
float laptimeThd = 60.0f;

int cycles = 10;
float mutationChance = 0.03f;

#define popSize 10
genann* population[popSize];
genann* inferenceNN = NULL;
bool popIsInitialized = false;
int fitness[popSize];
int maxFitness = 1;

int currentIndividual = 0;
int currentCycle = 0;


// neural network architecture
#define inputNeuronCnt 9
#define hiddenLayerCnt 0
#define hiddenNeuronCnt 0
#define outputNeuronCnt 3

// 0: random
// 1: prev
// 2: inference
int mode = 0;


/*
    todo:
    - maybe use same magnitude between input data
        e. g. distance sensor gives back 200
        divide by 10?
        reason to do so: after a certain speed
        the network becomes blind to any change
        ha 45 felett minden 1 lesz a sigmoid után
        akkor nem fog értesulni arrol hogy 100 megy vagy 50 nel

    - kivalasztani az inputokat
    - reward policy kitalalasa
    - feltetel hogy lealljon az egyed probalkozasa
    // possible addition to input:
    // z, trackpos


*/


// only for program flow exploration
int counter = 0;


//gives 19 angles for the distance sensors
void Cinit(float* angles)
{
    // init random generator
    srand(time(0));

    if (popIsInitialized)
    {
        for (int i = 0; i < popSize; i++)
            genann_free(population[i]);
        popIsInitialized = false;
    }


    if (mode == 0) // start from random
    {
        for (int i = 0; i < popSize; i++)
            population[i] = genann_init(inputNeuronCnt, hiddenLayerCnt, hiddenNeuronCnt, outputNeuronCnt);

        popIsInitialized = true;

        for (int i = 0; i < popSize; i++)
            fitness[i] = 1;
    }
    else if (mode == 1) // start from file
    {
        for (int i = 0; i < popSize; i++)
        {
            char path[10];
            sprintf(path, "%02d.txt", i);

            FILE* in = fopen(path, "r");
            population[i] = genann_read(in);
            fclose(in);
        }

        popIsInitialized = true;

        for (int i = 0; i < popSize; i++)
            fitness[i] = 1;
    }
    else if (mode == 2)
    {
        if (inferenceNN != NULL)
            genann_free(inferenceNN);

        FILE* in = fopen("10.txt", "r");
        inferenceNN = genann_read(in);
        fclose(in);
    }
        

    prevDistRaced = 0.0f;

    // set angles as {-90,-75,-60,-45,-30,20,15,10,5,0,5,10,15,20,30,45,60,75,90}

    for (int i = 0; i < 5; i++)
    {
        angles[i] = -90 + i * 15;
        angles[18 - i] = 90 - i * 15;
    }

    for (int i = 5; i < 9; i++)
    {
        angles[i] = -20 + (i - 5) * 5;
        angles[18 - i] = 20 - (i - 5) * 5;
    }
    angles[9] = 0;
}


void evaluate(structCarState cs)
{
    int points = 1;
    
    //if moves inside the track gets points;
    if (cs.trackPos > -1 && cs.trackPos < 1) {
        points += ((int)cs.distRaced - prevDistRaced)*4;
        printf("\npoints from moving inside:\t%d", ((int)cs.distRaced - (int)prevDistRaced) * 4);
    }
    else {
        points += ((int)cs.distRaced - prevDistRaced);
        printf("\npoints from moving outside:\t%d", (int)cs.distRaced - (int)prevDistRaced);
    }
    prevDistRaced = cs.distRaced;
    

    float dmgMultiplier = 1.0f;
    if (cs.damage != prevDamage) {
        points += (int)(prevDamage - cs.damage) * dmgMultiplier;
        prevDamage = cs.damage;
        printf("\npoints from damage:\t%d", (int)((prevDamage - cs.damage) * dmgMultiplier));
    }

        
    fitness[currentIndividual] += points;
    if (fitness[currentIndividual] < 1)
        fitness[currentIndividual] = 1;


    if (fitness[currentIndividual] > maxFitness)
        maxFitness = fitness[currentIndividual];
}

genann* acceptReject()
{
    int counter = 0;
    int safetyThd = 100;
    while (42)
    {
        int parentIdx = rand() % popSize;
        if (fitness[parentIdx] > rand() % maxFitness)
            return population[parentIdx];

        if (counter > safetyThd)
            return population[parentIdx];

        counter++;
    }
}

void crossover()
{
    genann* new_pop[popSize];
    int weightCnt = population[0]->total_weights;

    for (int i = 0; i < popSize; i += 2)
    {
        genann* p1 = acceptReject();
        genann* p2 = acceptReject();


        genann* c1 = genann_init(inputNeuronCnt, hiddenLayerCnt, hiddenNeuronCnt, outputNeuronCnt);
        genann* c2 = genann_init(inputNeuronCnt, hiddenLayerCnt, hiddenNeuronCnt, outputNeuronCnt);
        new_pop[i] = c1;
        new_pop[i + 1] = c2;

        // crossover
        int crossOverPoint = rand() % weightCnt;
        for (int j = 0; j < weightCnt; j++)
        {
            if (j < crossOverPoint)
            {
                c1->weight[j] = p1->weight[j];
                c2->weight[j] = p2->weight[j];
            }
            else
            {
                c1->weight[j] = p2->weight[j];
                c2->weight[j] = p1->weight[j];
            }
        }

        // mutation
        if ((float)rand() < mutationChance)
        {
            int weightIdx = rand() % weightCnt;
            float randFloat = (float)rand();

            c1->weight[weightIdx] = randFloat;
            c2->weight[weightCnt - 1 - weightIdx] = randFloat;
        }
    }


    for (int i = 0; i < popSize; i++)
    {
        genann_free(population[i]);
        population[i] = genann_copy(new_pop[i]);
        genann_free(new_pop[i]);
    }
}


void next()
{
    if (currentIndividual == popSize - 1)
    {
        if (currentCycle == cycles - 1)
        {
            //stop and save the best individual to a file
            time_t t = time(NULL);
            struct tm tm = *localtime(&t);

            int idx[popSize];
            for (int i = 0; i < popSize; i++)
                idx[i] = i;

            // sort fitness in descending order
            for (int i = 0; i < popSize - 1; i++)
            {
                for (int j = i + 1; j < popSize; j++)
                {
                    if (fitness[idx[j]] > fitness[idx[i]])
                    {
                        int temp = idx[i];
                        idx[i] = idx[j];
                        idx[j] = temp;
                    }
                }
            }

            for (int i = 0; i < popSize; i++)
            {
                char path[10];
                sprintf(path, "%02d.txt", i);
                FILE* out = fopen(path, "w");
                genann_write(population[idx[i]], out);
                fclose(out);
            }


            exit(0);
        }

        crossover();
        currentIndividual = 0;
        currentCycle++;
        return;
    }

    currentIndividual++;
}

structCarControl CDrive(structCarState cs)
{

    clutching(&cs, &clutch);
    int gear = getGear(&cs);
    int meta = 0;

    
    double input[inputNeuronCnt];
    input[0] = (double)cs.track[1] / 4;
    input[1] = (double)cs.track[5] / 4;
    input[2] = (double)cs.track[9] / 4;
    input[3] = (double)cs.track[13] / 4;
    input[4] = (double)cs.track[17] / 4;
    input[5] = (double)cs.angle * 10; // cs.angle [-3,14, +3,14] in radian
https://towardsdatascience.com/17-rules-of-thumb-for-building-a-neural-network-93356f9930af
    input[6] = (double)cs.trackPos;
    input[7] = (double)cs.speedX;
    input[8] = (double)cs.speedY;

    //printf("\n\ninputs:");
    //for (int i = 0; i < inputNeuronCnt; i++)
    //    printf("\ninput_%02d:\t%f", i, input[i]);

    

    double* prediction;
    if (mode == 0 || mode == 1)
        prediction = genann_run(population[currentIndividual], input);
    else
        prediction = genann_run(inferenceNN, input);

    double accel = prediction[0];
    double brake = prediction[1];
    double steer = prediction[2] * 2 - 1;

    //printf("\n\outputs:");
    //printf("\naccel:\t%f", accel);
    //printf("\nbrake:\t%f", brake);
    //printf("\nsteer:\t%f", steer);
    
    if (accel > brake)
        brake = 0;
    else
        accel = 0;

    if (mode == 0 || mode == 1)
    {
        evaluate(cs);
        if (cs.curLapTime > laptimeThd)
        {
            meta = 1;
            next();
        }
    }

    printf("\nfitness:\t%d", fitness[currentIndividual]);
    structCarControl cc = { accel, brake, gear, steer, clutch, meta };
    return cc;
}


void ConShutdown()
{
    printf("Bye bye!");
}

void ConRestart()
{
    prevDistRaced = 0.0f;
    maxFitness = 1;

    printf("\n\ncounter:\t%d\n\n", counter);
    counter += 1;
    printf("Restarting the race!");
}


int getGear(structCarState* cs)
{

    int gear = cs->gear;
    int rpm = cs->rpm;

    // if gear is 0 (N) or -1 (R) just return 1 
    if (gear < 1)
        return 1;
    // check if the RPM value of car is greater than the one suggested 
    // to shift up the gear from the current one     
    if (gear < 6 && rpm >= gearUp[gear - 1])
        return gear + 1;
    else
        // check if the RPM value of car is lower than the one suggested 
        // to shift down the gear from the current one
        if (gear > 1 && rpm <= gearDown[gear - 1])
            return gear - 1;
        else // otherwhise keep current gear
            return gear;
}


void clutching(structCarState* cs, float* clutch)
{
    float maxClutch = clutchMax;

    // Check if the current situation is the race start
    if (cs->curLapTime < clutchDeltaTime && cs->stage == RACE && cs->distRaced < clutchDeltaRaced)
        *clutch = maxClutch;

    // Adjust the current value of the clutch
    if (clutch > 0)
    {
        float delta = clutchDelta;
        if (cs->gear < 2)
        {
            // Apply a stronger clutch output when the gear is one and the race is just started
            delta /= 2;
            maxClutch *= clutchMaxModifier;
            if (cs->curLapTime < clutchMaxTime)
                *clutch = maxClutch;
        }

        // check clutch is not bigger than maximum values
        *clutch = fmin(maxClutch, *clutch);

        // if clutch is not at max value decrease it quite quickly
        if (*clutch != maxClutch)
        {
            *clutch -= delta;
            *clutch = fmax(0.0, *clutch);
        }
        // if clutch is at max value decrease it very slowly
        else
            *clutch -= clutchDec;
    }
}