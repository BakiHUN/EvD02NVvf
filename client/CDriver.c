#include "CDriver.h"
// "genann.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015-2018 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#include "genann.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef genann_act
#define genann_act_hidden genann_act_hidden_indirect
#define genann_act_output genann_act_output_indirect
#else
#define genann_act_hidden genann_act
#define genann_act_output genann_act
#endif

#define LOOKUP_SIZE 4096

double genann_act_hidden_indirect(const struct genann* ann, double a) {
    return ann->activation_hidden(ann, a);
}

double genann_act_output_indirect(const struct genann* ann, double a) {
    return ann->activation_output(ann, a);
}

const double sigmoid_dom_min = -15.0;
const double sigmoid_dom_max = 15.0;
double interval;
double lookup[LOOKUP_SIZE];

#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#define unused          __attribute__((unused))
#else
#define likely(x)       x
#define unlikely(x)     x
#define unused
#pragma warning(disable : 4996) /* For fscanf */
#endif


double genann_act_sigmoid(const genann* ann unused, double a) {
    if (a < -45.0) return 0;
    if (a > 45.0) return 1;
    return 1.0 / (1 + exp(-a));
}

void genann_init_sigmoid_lookup(const genann* ann) {
    const double f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;
    int i;

    interval = LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);
    for (i = 0; i < LOOKUP_SIZE; ++i) {
        lookup[i] = genann_act_sigmoid(ann, sigmoid_dom_min + f * i);
    }
}

double genann_act_sigmoid_cached(const genann* ann unused, double a) {
    assert(!isnan(a));

    if (a < sigmoid_dom_min) return lookup[0];
    if (a >= sigmoid_dom_max) return lookup[LOOKUP_SIZE - 1];

    size_t j = (size_t)((a - sigmoid_dom_min) * interval + 0.5);

    /* Because floating point... */
    if (unlikely(j >= LOOKUP_SIZE)) return lookup[LOOKUP_SIZE - 1];

    return lookup[j];
}

double genann_act_linear(const struct genann* ann unused, double a) {
    return a;
}

double genann_act_threshold(const struct genann* ann unused, double a) {
    return a > 0;
}

genann* genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
    if (hidden_layers < 0) return 0;
    if (inputs < 1) return 0;
    if (outputs < 1) return 0;
    if (hidden_layers > 0 && hidden < 1) return 0;


    const int hidden_weights = hidden_layers ? (inputs + 1) * hidden + (hidden_layers - 1) * (hidden + 1) * hidden : 0;
    const int output_weights = (hidden_layers ? (hidden + 1) : (inputs + 1)) * outputs;
    const int total_weights = (hidden_weights + output_weights);

    const int total_neurons = (inputs + hidden * hidden_layers + outputs);

    /* Allocate extra size for weights, outputs, and deltas. */
    const int size = sizeof(genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    genann* ret = malloc(size);
    if (!ret) return 0;

    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;

    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;

    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    genann_randomize(ret);

    ret->activation_hidden = genann_act_sigmoid_cached;
    ret->activation_output = genann_act_sigmoid_cached;

    genann_init_sigmoid_lookup(ret);

    return ret;
}


genann* genann_read(FILE* in) {
    int inputs, hidden_layers, hidden, outputs;
    int rc;

    errno = 0;
    rc = fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);
    if (rc < 4 || errno != 0) {
        perror("fscanf");
        return NULL;
    }

    genann* ann = genann_init(inputs, hidden_layers, hidden, outputs);

    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        errno = 0;
        rc = fscanf(in, " %le", ann->weight + i);
        if (rc < 1 || errno != 0) {
            perror("fscanf");
            genann_free(ann);

            return NULL;
        }
    }

    return ann;
}


genann* genann_copy(genann const* ann) {
    const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
    genann* ret = malloc(size);
    if (!ret) return 0;

    memcpy(ret, ann, size);

    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    return ret;
}


void genann_randomize(genann* ann) {
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        double r = GENANN_RANDOM();
        /* Sets weights from -0.5 to 0.5. */
        ann->weight[i] = r - 0.5;
    }
}


void genann_free(genann* ann) {
    /* The weight, output, and delta pointers go to the same buffer. */
    free(ann);
}


double const* genann_run(genann const* ann, double const* inputs) {
    double const* w = ann->weight;
    double* o = ann->output + ann->inputs;
    double const* i = ann->output;

    /* Copy the inputs to the scratch area, where we also store each neuron's
     * output, for consistency. This way the first layer isn't a special case. */
    memcpy(ann->output, inputs, sizeof(double) * ann->inputs);

    int h, j, k;

    if (!ann->hidden_layers) {
        double* ret = o;
        for (j = 0; j < ann->outputs; ++j) {
            double sum = *w++ * -1.0;
            for (k = 0; k < ann->inputs; ++k) {
                sum += *w++ * i[k];
            }
            *o++ = genann_act_output(ann, sum);
        }

        return ret;
    }

    /* Figure input layer */
    for (j = 0; j < ann->hidden; ++j) {
        double sum = *w++ * -1.0;
        for (k = 0; k < ann->inputs; ++k) {
            sum += *w++ * i[k];
        }
        *o++ = genann_act_hidden(ann, sum);
    }

    i += ann->inputs;

    /* Figure hidden layers, if any. */
    for (h = 1; h < ann->hidden_layers; ++h) {
        for (j = 0; j < ann->hidden; ++j) {
            double sum = *w++ * -1.0;
            for (k = 0; k < ann->hidden; ++k) {
                sum += *w++ * i[k];
            }
            *o++ = genann_act_hidden(ann, sum);
        }

        i += ann->hidden;
    }

    double const* ret = o;

    /* Figure output layer. */
    for (j = 0; j < ann->outputs; ++j) {
        double sum = *w++ * -1.0;
        for (k = 0; k < ann->hidden; ++k) {
            sum += *w++ * i[k];
        }
        *o++ = genann_act_output(ann, sum);
    }

    /* Sanity check that we used all weights and wrote all outputs. */
    assert(w - ann->weight == ann->total_weights);
    assert(o - ann->output == ann->total_neurons);

    return ret;
}


void genann_train(genann const* ann, double const* inputs, double const* desired_outputs, double learning_rate) {
    /* To begin with, we must run the network forward. */
    genann_run(ann, inputs);

    int h, j, k;

    /* First set the output layer deltas. */
    {
        double const* o = ann->output + ann->inputs + ann->hidden * ann->hidden_layers; /* First output. */
        double* d = ann->delta + ann->hidden * ann->hidden_layers; /* First delta. */
        double const* t = desired_outputs; /* First desired output. */


        /* Set output layer deltas. */
        if (genann_act_output == genann_act_linear ||
            ann->activation_output == genann_act_linear) {
            for (j = 0; j < ann->outputs; ++j) {
                *d++ = *t++ - *o++;
            }
        }
        else {
            for (j = 0; j < ann->outputs; ++j) {
                *d++ = (*t - *o) * *o * (1.0 - *o);
                ++o; ++t;
            }
        }
    }


    /* Set hidden layer deltas, start on last layer and work backwards. */
    /* Note that loop is skipped in the case of hidden_layers == 0. */
    for (h = ann->hidden_layers - 1; h >= 0; --h) {

        /* Find first output and delta in this layer. */
        double const* o = ann->output + ann->inputs + (h * ann->hidden);
        double* d = ann->delta + (h * ann->hidden);

        /* Find first delta in following layer (which may be hidden or output). */
        double const* const dd = ann->delta + ((h + 1) * ann->hidden);

        /* Find first weight in following layer (which may be hidden or output). */
        double const* const ww = ann->weight + ((ann->inputs + 1) * ann->hidden) + ((ann->hidden + 1) * ann->hidden * (h));

        for (j = 0; j < ann->hidden; ++j) {

            double delta = 0;

            for (k = 0; k < (h == ann->hidden_layers - 1 ? ann->outputs : ann->hidden); ++k) {
                const double forward_delta = dd[k];
                const int windex = k * (ann->hidden + 1) + (j + 1);
                const double forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
            }

            *d = *o * (1.0 - *o) * delta;
            ++d; ++o;
        }
    }


    /* Train the outputs. */
    {
        /* Find first output delta. */
        double const* d = ann->delta + ann->hidden * ann->hidden_layers; /* First output delta. */

        /* Find first weight to first output delta. */
        double* w = ann->weight + (ann->hidden_layers
            ? ((ann->inputs + 1) * ann->hidden + (ann->hidden + 1) * ann->hidden * (ann->hidden_layers - 1))
            : (0));

        /* Find first output in previous layer. */
        double const* const i = ann->output + (ann->hidden_layers
            ? (ann->inputs + (ann->hidden) * (ann->hidden_layers - 1))
            : 0);

        /* Set output layer weights. */
        for (j = 0; j < ann->outputs; ++j) {
            *w++ += *d * learning_rate * -1.0;
            for (k = 1; k < (ann->hidden_layers ? ann->hidden : ann->inputs) + 1; ++k) {
                *w++ += *d * learning_rate * i[k - 1];
            }

            ++d;
        }

        assert(w - ann->weight == ann->total_weights);
    }


    /* Train the hidden layers. */
    for (h = ann->hidden_layers - 1; h >= 0; --h) {

        /* Find first delta in this layer. */
        double const* d = ann->delta + (h * ann->hidden);

        /* Find first input to this layer. */
        double const* i = ann->output + (h
            ? (ann->inputs + ann->hidden * (h - 1))
            : 0);

        /* Find first weight to this layer. */
        double* w = ann->weight + (h
            ? ((ann->inputs + 1) * ann->hidden + (ann->hidden + 1) * (ann->hidden) * (h - 1))
            : 0);


        for (j = 0; j < ann->hidden; ++j) {
            *w++ += *d * learning_rate * -1.0;
            for (k = 1; k < (h == 0 ? ann->inputs : ann->hidden) + 1; ++k) {
                *w++ += *d * learning_rate * i[k - 1];
            }
            ++d;
        }

    }

}


void genann_write(genann const* ann, FILE* out) {
    fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);

    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        fprintf(out, " %.20e", ann->weight[i]);
    }
}


bool dummy = true;
int meanFit = 0;
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
float laptimeThd = 180.0f;

int cycles = 999;
float mutationChance = 0.30f;

#define popSize 10
genann* population[popSize];
genann* inferenceNN = NULL;
bool popIsInitialized = false;
int fitness[popSize];
int maxFitness = 1;
bool isFintessInitid = false;

int currentIndividual = 0;
int currentCycle = 0;

int stuck = 0;
int maxStuck = 300;

float prevCurLapTime = -10.0f;
int lapsCompleted = 0;

int bestIdx=0;

// neural network architecture
#define inputNeuronCnt 9
#define hiddenLayerCnt 1
#define hiddenNeuronCnt 6
#define outputNeuronCnt 3

// 0: random
// 1: prev
// 2: inference
int mode = 2;
const char* crossover_log_path = "crossover_log.txt";


/*
    todo:
    - maybe use same magnitude between input data
        e. g. distance sensor gives back 200
        divide by 10?
        reason to do so: after a certain speed
        the network becomes blind to any change
        ha 45 felett minden 1 lesz a sigmoid ut�n
        akkor nem fog ertesulni arrol hogy 100 megy vagy 50 nel

    - kivalasztani az inputokat
    - reward policy kitalalasa
    - feltetel hogy lealljon az egyed probalkozasa
    // possible addition to input:
    // z, trackpos
*/



//gives 19 angles for the distance sensors
float RandomFloat(float a, float b) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}
void Cinit(float* angles)
{
    if (!isFintessInitid)
    {
        for (int i = 0; i < popSize; i++)
            fitness[i] = 1;
        
        // init random generator
        srand(time(0));
        isFintessInitid = true;
    }

    if (mode == 0 && dummy) // start from random
    {
        //printf("\n\nASDASDASDASDASDASDAS\n\n");
        for (int i = 0; i < popSize; i++)
            population[i] = genann_init(inputNeuronCnt, hiddenLayerCnt, hiddenNeuronCnt, outputNeuronCnt);

        popIsInitialized = true;
        dummy = false;
    }
    else if (mode == 1 && dummy) // start from file
    {
        for (int i = 0; i < popSize; i++)
        {
            FILE* in = fopen("ferenc.txt", "r");
            population[i] = genann_read(in);
            fclose(in);
        }
        
        popIsInitialized = true;
        dummy = false;
    }
    else if (mode == 2) //inference
    {
        if (inferenceNN == NULL)
        {
            FILE* in = fopen("ferenc.txt", "r");
            inferenceNN = genann_read(in);
            fclose(in);
        }    
    }        

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
    int points = 0;
    float distDiff = (cs.distRaced - prevDistRaced);
    //printf("\ndistRaced:\t%02f", cs.distRaced);
    if ((cs.distRaced - prevDistRaced) <= 0.001f)
        stuck++;
    else
        stuck = 0;
    //printf("\tStuck: %d", stuck);
    //printf("\tasd: %f\n", distDiff);

    int multiplier = 10;
    if (cs.distRaced > 0.0f && cs.trackPos > -1 && cs.trackPos < 1)
        multiplier = 14;

    points += (int)(distDiff * multiplier);
    prevDistRaced = cs.distRaced;
    //damage punishment
    float dmgMultiplier = 1.0f;
    if (cs.damage != prevDamage) {
        points += (int)(prevDamage - cs.damage) * dmgMultiplier;
        prevDamage = cs.damage;
        //printf("\npoints from damage:\t%d", (int)((prevDamage - cs.damage) * dmgMultiplier));
    }

    //printf("\npoints:\t%d", points);
    fitness[currentIndividual] += points;
    if (fitness[currentIndividual] < 1)
        fitness[currentIndividual] = 1;


    if (fitness[currentIndividual] > maxFitness)
        maxFitness = fitness[currentIndividual];
}


void crossover()
{
    genann* new_pop[popSize];
    int weightCnt = population[0]->total_weights;
    new_pop[0] = genann_copy(population[bestIdx]);
    for (int i = 1; i < popSize; i++)
    {
        new_pop[i] = genann_copy(new_pop[0]);
        for (int j = 0; j < weightCnt; j++) {
            double mutation = RandomFloat(-0.05, 0.05);
            if ((float)rand() / RAND_MAX < mutationChance && new_pop[i]->weight[j] + mutation > -0.5f && new_pop[i]->weight[j] + mutation < 0.5)
                new_pop[i]->weight[j] += mutation;
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
    prevDamage = 0.0f;
    prevDistRaced = 0.0f;
    lapsCompleted = 0;
    prevCurLapTime = -10.0f;
    meanFit = 0;

    if (currentIndividual == popSize - 1)
    {
        bestIdx = 0;
        for (int i = 0; i < popSize; i++) {
            if (fitness[i] > fitness[bestIdx])
                bestIdx = i;
        }
        printf("\nNEXT CYCLE:\t%2d", currentCycle);
        
        FILE* fp;
        fp = fopen(crossover_log_path, "a");
        char gen[200];
        sprintf(gen, "\n\n\nGeneration: %d\nBestIdx: %d\nFitness of bestIdx: %d\n",currentCycle,bestIdx,fitness[bestIdx]);
        fputs(gen, fp);
        fputs("\nfitness values", fp);

        for (int i = 0; i < popSize; i++)
        {
            meanFit += fitness[i];
            char data[200];
            sprintf(data, "\n%d:\tfitness:\t%d", i, fitness[i]);
            fputs(data, fp);
        }

        char data[200];
        meanFit /= popSize;
        sprintf(data, "\nMean fitness:\t%d", meanFit);
        fputs(data, fp);
        fclose(fp);

        //Printing every generation
        char dir[100];
        struct stat st = { 0 };
        sprintf(dir, "gen%03d", currentCycle);
        if (stat(dir, &st) == -1) {
            mkdir(dir, 0777);
        }
        for (int i = 0; i < popSize; i++)
        {
            char path[100];
            sprintf(path, "gen%03d/%02d.txt", currentCycle, i);
            FILE* out = fopen(path, "w");
            genann_write(population[i], out);
            fclose(out);
        }


        if (currentCycle == cycles - 1)
        {
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

            FILE* fp;
            fp = fopen(crossover_log_path, "a");
            fputs("\n\nidx array", fp);
            for (int i = 0; i < popSize; i++)
            {
                char data[200];
                sprintf(data, "\nidx[%d]:\t%d\t\tfitness:\t%d", i, idx[i], fitness[idx[i]]);
                fputs(data, fp);
            }
            fclose(fp);
          
            if (popIsInitialized)
            {
                for (int i = 0; i < popSize; i++)
                    genann_free(population[i]);
                popIsInitialized = false;

            }

            exit(11);
        }
        crossover();
        currentIndividual = 0;
        currentCycle++;
        maxFitness = 1;

        for (int i = 0; i < popSize; i++)
            fitness[i] = 1;

        return 69;
    }

    currentIndividual++;
}

structCarControl CDrive(structCarState cs)
{
    //printf("\n%f", cs.distFromStart);
    //printf("\n%f", cs.curLapTime);

    if (cs.curLapTime < prevCurLapTime)
        lapsCompleted++;
    prevCurLapTime = cs.curLapTime;
    //printf("\n%d", lapsCompleted);


    clutching(&cs, &clutch);
    int gear = getGear(&cs);
    int meta = 2;

    
    //https://towardsdatascience.com/17-rules-of-thumb-for-building-a-neural-network-93356f9930af
    double input[inputNeuronCnt];
    input[0] = (double)cs.track[1] / 4;
    input[1] = (double)cs.track[5] / 4;
    input[2] = (double)cs.track[9] / 4;
    input[3] = (double)cs.track[13] / 4;
    input[4] = (double)cs.track[17] / 4;
    input[5] = (double)cs.angle * 10; // cs.angle [-3,14, +3,14] in radian
    input[6] = (double)cs.trackPos * 10;
    input[7] = (double)cs.speedX / 5;
    input[8] = (double)cs.speedY;

    //printf("\n\ninputs:");
    //for (int i = 0; i < inputNeuronCnt; i++)
    //    printf("\ninput_%02d:\t%f", i, input[i]);

    
    const double* prediction;
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
        if (cs.curLapTime > laptimeThd || stuck > maxStuck || lapsCompleted > 2)
        {
            meta = 1;
            next();
            stuck = 0;
        }
    }

    //printf("\n gen: %d fitness %02d:\t%d", currentCycle, currentIndividual, fitness[currentIndividual]);
    //printf("\nlongitudnal speed:\t%f", input[7]);
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
    prevDamage = 0.0f;
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