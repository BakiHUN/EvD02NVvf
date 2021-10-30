#include "CDriver.h"
// "genann.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <string.h>
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
#ifndef GENANN_H
#define GENANN_H



#ifdef __cplusplus
extern "C" {
#endif

#ifndef GENANN_RANDOM
    /* We use the following for uniform random numbers between 0 and 1.
     * If you have a better function, redefine this macro. */
#define GENANN_RANDOM() (((double)rand())/RAND_MAX)
#endif

    struct genann;

    typedef double (*genann_actfun)(const struct genann* ann, double a);

    typedef struct genann {
        /* How many inputs, outputs, and hidden neurons. */
        int inputs, hidden_layers, hidden, outputs;

        /* Which activation function to use for hidden neurons. Default: gennann_act_sigmoid_cached*/
        genann_actfun activation_hidden;

        /* Which activation function to use for output. Default: gennann_act_sigmoid_cached*/
        genann_actfun activation_output;

        /* Total number of weights, and size of weights buffer. */
        int total_weights;

        /* Total number of neurons + inputs and size of output buffer. */
        int total_neurons;

        /* All weights (total_weights long). */
        double* weight;

        /* Stores input array and output of each neuron (total_neurons long). */
        double* output;

        /* Stores delta of each hidden and output neuron (total_neurons - inputs long). */
        double* delta;

    } genann;

    /* Creates and returns a new ann. */
    genann* genann_init(int inputs, int hidden_layers, int hidden, int outputs);

    /* Creates ANN from file saved with genann_write. */
    genann* genann_read(FILE* in);

    /* Sets weights randomly. Called by init. */
    void genann_randomize(genann* ann);

    /* Returns a new copy of ann. */
    genann* genann_copy(genann const* ann);

    /* Frees the memory used by an ann. */
    void genann_free(genann* ann);

    /* Runs the feedforward algorithm to calculate the ann's output. */
    double const* genann_run(genann const* ann, double const* inputs);

    /* Does a single backprop update. */
    void genann_train(genann const* ann, double const* inputs, double const* desired_outputs, double learning_rate);

    /* Saves the ann. */
    void genann_write(genann const* ann, FILE* out);

    void genann_init_sigmoid_lookup(const genann* ann);
    double genann_act_sigmoid(const genann* ann, double a);
    double genann_act_sigmoid_cached(const genann* ann, double a);
    double genann_act_threshold(const genann* ann, double a);
    double genann_act_linear(const genann* ann, double a);


#ifdef __cplusplus
}
#endif

#endif /*GENANN_H*/


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

double weights[] = {
4.33320108273418713729e-01,
1.67212437564777105514e-01,
-1.12720114749595645698e-01,
3.46354539914143222390e-02,
-8.59439374138338818554e-02,
2.07399213449143249299e-01,
-2.61555526950554506627e-01,
8.11182097861129491889e-03,
-2.88272801645344078914e-01,
-2.64676349946590838691e-01,
-1.47900324026514040643e-01,
4.70599841533583362896e-01,
-2.54325998718222590078e-01,
-4.00921623444083774501e-02,
1.03170869524479513757e-01,
-3.07123021066923684863e-01,
-3.89660359249571763840e-02,
-2.61121249522879794736e-01,
-2.74405799624980906160e-01,
4.20343644842942065765e-01,
-8.18639188481297130906e-02,
-3.90098269013347764833e-01,
2.99955289174329675639e-01,
-3.18898587307410441571e-01,
6.08111802242653487482e-02,
3.97343369077983288307e-01,
-2.04104738984674571878e-01,
3.43029571598830629320e-01,
4.00756865619538049650e-01,
3.71852778710287767971e-01,
-2.12782826934975133337e-01,
1.99586477717293431233e-01,
2.45410015560359351383e-01,
3.43709375072189016187e-01,
8.58287313834178444694e-02,
-3.76888332773827328115e-01,
3.07275612659077768107e-01,
3.11381421382971423917e-01,
4.80864894558549749171e-01,
6.48953221546713088230e-02,
-4.93475143792721671065e-01,
-4.12920008297614771209e-01,
4.49082918363270966466e-01,
1.51547284592020936600e-01,
-1.26514481032746373135e-01,
3.75255440204080970013e-01,
-3.21955015717032388967e-01,
-2.63145848935659343315e-01,
4.25206763570312618050e-01,
-8.20825803667606379577e-02,
2.24295785393841318189e-01,
2.95390178292587735065e-01,
4.74332407978094061640e-01,
4.24470501691359025642e-01,
1.32929470063449128858e-01,
-2.06564531388287009328e-01,
-5.72879748122138710009e-02,
4.63112735151922860766e-01,
8.83419267648145600802e-02,
-1.16916387257075538031e-02,
1.12079220493531139802e-01,
-4.20177313420564246815e-01,
3.58567612128549484396e-01,
1.94389629394627672099e-01,
-2.22429577892986507504e-01,
-4.86205633716849272563e-01,
-2.99715110842180276762e-01,
-9.28830835901974793956e-02,
-2.32026124134953992684e-01,
-2.24219490778665431208e-02,
-8.42478746465812888822e-02,
-2.88724937261428338253e-01,
-3.00836207159642321507e-01,
-2.66249429675177617938e-01,
-4.78484450819421980317e-01,
-3.05914330104735454618e-01,
3.43043303904177143693e-01,
9.85976732913957931714e-02,
-7.70836478343604469821e-02,
4.55301824905488539130e-01,
-4.77806937850714597005e-01,
-1.10388502091995632792e-01,
1.63878902736163567511e-01,
-4.78346048837721082858e-01,
3.23767964567560628808e-01,
2.96790982380195189627e-01,
-4.11392562959944529943e-01,
1.59933775313096049331e-01,
1.73887753057944127733e-01,
3.60708943455815611578e-01,
2.58470413060103099134e-01,
-1.34707174857329525786e-01,
-4.74635610799594320763e-01,
3.73226129588180022267e-02,
1.66493732206723255018e-01,
-6.07249667919121938198e-02,
-1.64845728934598811932e-01,
1.72014525736319034976e-01,
-1.94280831779414242533e-02,
2.29369485530292016584e-01,
3.61960205130767209702e-01,
-5.10376277528483690560e-02,
-2.65570546241256200126e-01,
-3.78864408313257627903e-01,
1.88717917257227307815e-01,
4.74825281601282234156e-01,
3.57856991588473616339e-01,
4.20872373010950262540e-01,
3.40800202221963499660e-01,
1.56775112861158749666e-01,
1.20050206362451228337e-01,
1.89735103102398272590e-01,
-6.40882678412002526613e-04,
3.40838189543371417045e-01,
-9.18973391888642243686e-02,
3.48336750454956267653e-01,
-3.87203596038019348669e-02,
2.76041443336420888599e-01,
-2.92194736734325999361e-01,
-2.72115239733973668379e-01,
8.67015566228573897334e-02,
3.41920377745390524638e-01,
4.56663177847460577397e-03,
1.49615462522650344290e-01,
2.42401345859228234403e-01,
-2.87743766594439565054e-01,
3.73609118986589772149e-01,
2.56782756712981008462e-02,
-3.05240027419068804537e-01,
1.05151524399548335076e-01,
3.65030364967762310791e-01,
-6.18259231884376037058e-02,
-4.19423504651800671539e-01,
2.88310040973322356805e-01,
4.92759481956822731341e-01,
8.77452963618043924043e-02,
-2.52029482047071096140e-01,
-2.78646196930587342067e-01,
1.41689809665441313058e-01,
2.11076689597623823325e-01,
8.35443984043060527966e-03,
-3.20473342020329843294e-01,
7.00766002561459266929e-02,
-4.73079623614522604136e-01,
6.06707623562396936023e-03,
4.15971863068164804389e-01,
-3.28708761190989795509e-01
};


// CUSTOM STUFF FROM HERE
static struct
{
    int cycles;
    float mutationChance;
    float mutationLimit;
    bool popIsInitialized;
    int totalLaps;
    int curTry;
    int curIndividuum;
    int curCycle;
} GA = 
    { 
     .cycles = 999,
     .mutationChance = 0.05f,
     .mutationLimit = 0.015f,
     .popIsInitialized = false,
     .totalLaps = 3,
     .curTry = 0,
     .curIndividuum = 0,
     .curCycle = 0
    };


static struct
{
    float laptimeThd;
    float dmgMultiplier;
    float onTrackMultiplier;
    float offTrackMultiplier;
} RewardPolicy =
    {
     .laptimeThd = 180.0f,
     .dmgMultiplier = 1.0f,
     .onTrackMultiplier = 1.0f,
     .offTrackMultiplier = 0.01f     
    };


#define totalTries 2
#define popSize 20
#define inputNeuronCnt 10
#define hiddenLayerCnt 1
#define hiddenNeuronCnt 8
#define outputNeuronCnt 3
#define top 5

#define S 5
float sensorHistory1[S];
float sensorHistory5[S];
float sensorHistory9[S];
float sensorHistory13[S];
float sensorHistory17[S];


enum Mode { train_random, train_continue, inference };
enum Mode mode = train_random;

genann* population[popSize];
genann* inferenceNN = NULL;
float fitness[popSize];
float fitnessPerTry[totalTries];
float closest;

bool isFitnessInitid = false;
int stuck = 0;
int maxStuck = 300;
float prevCurLapTime = -10.0f;
int lapsCompleted = 0;
int bestIdx = 0;
float prevDamage = 0.0f;
float prevDistRaced = 0.0f;

const char* logPath = "crossover_log.txt";



//gives 19 angles for the distance sensors
float RandomFloat(float a, float b) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

void InitLogFile()
{
    FILE* f;
    f = fopen(logPath, "a");
    char data[5000];

    sprintf(data, "mutation chance:\t%f", GA.mutationChance);
    fputs(data, f);
    sprintf(data, "\nmutation limit:\t%f", GA.mutationLimit);
    fputs(data, f);
    sprintf(data, "\nhidden layer count:\t%d", hiddenLayerCnt);
    fputs(data, f);
    sprintf(data, "\nhidden neuron count:\t%d", hiddenNeuronCnt);
    fputs(data, f);
    sprintf(data, "\npop size:\t%d", popSize);
    fputs(data, f);
    sprintf(data, "\ntotal tries:\t%d", totalTries);
    fputs(data, f);
    sprintf(data, "\non track multiplier:\t%f", RewardPolicy.onTrackMultiplier);
    fputs(data, f);
    sprintf(data, "\noff track multiplier:\t%f", RewardPolicy.offTrackMultiplier);
    fputs(data, f);
    sprintf(data, "\ndamage multiplier:\t%f", RewardPolicy.dmgMultiplier);
    fputs(data, f);
    sprintf(data, "\nlaptime threshold:\t%f", RewardPolicy.laptimeThd);        
    fputs(data, f);
    sprintf(data, "\ntotal laps:\t%d", GA.totalLaps);
    fputs(data, f);
    fclose(f);
}


void Cinit(float* angles)
{
    if (!isFitnessInitid)
    {
        for (int i = 0; i < popSize; i++)
            fitness[i] = 1;
        for (int i = 0; i < totalTries; i++)
            fitnessPerTry[i] = 1;

        // init random generator
        srand(time(0));
        isFitnessInitid = true;
        if(mode != inference)
            InitLogFile();

        for (int i = 0; i < S; i++)
        {
            sensorHistory1[i] = 0.0f;
            sensorHistory5[i] = 0.0f;
            sensorHistory9[i] = 0.0f;
            sensorHistory13[i] = 0.0f;
            sensorHistory17[i] = 0.0f;
        }
    }

    if (mode == train_random && !GA.popIsInitialized)
    {
        for (int i = 0; i < popSize; i++)
            population[i] = genann_init(inputNeuronCnt, hiddenLayerCnt, hiddenNeuronCnt, outputNeuronCnt);

        GA.popIsInitialized = true;
    }
    else if (mode == train_continue && !GA.popIsInitialized)
    {
        for (int i = 0; i < popSize; i++)
        {
            FILE* in = fopen("00.txt", "r");
            population[i] = genann_read(in);
            fclose(in);
        }

        GA.popIsInitialized = true;
    }
    else if (mode == inference)
    {
        if (inferenceNN == NULL)
        {
            inferenceNN = genann_init(inputNeuronCnt, hiddenLayerCnt, hiddenNeuronCnt, outputNeuronCnt);
            for (int i = 0; i < sizeof(weights) / sizeof(weights[0]); i++)
                inferenceNN->weight[i] = weights[i];
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
    float points = 0;

    float distCovered = (cs.distRaced - prevDistRaced);
    if (distCovered <= 0.001f)
        stuck++;
    else
        stuck = 0;

    if (distCovered > 0.0f && cs.trackPos > -1 && cs.trackPos < 1)
    {
        distCovered *= RewardPolicy.onTrackMultiplier;
        if(cs.speedX > 3.0f)
            distCovered *= log(cs.speedX);
    }
    else
        distCovered *= RewardPolicy.offTrackMultiplier;

    points += distCovered;
    prevDistRaced = cs.distRaced;

    //damage punishment
    if (cs.damage != prevDamage) {
        points += (prevDamage - cs.damage) * RewardPolicy.dmgMultiplier;
        prevDamage = cs.damage;
    }

    if (cs.curLapTime >= 0.0f) {
        fitnessPerTry[GA.curTry] += points;
        if (fitnessPerTry[GA.curTry] < 1)
            fitnessPerTry[GA.curTry] = 1;
    }
    else
        fitnessPerTry[GA.curTry] = 1.0f;
}


void reproduce()
{    
    genann* new_pop[popSize];
    int weightCnt = population[0]->total_weights;
    new_pop[0] = genann_copy(population[bestIdx]);

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

    for (int i = 1; i < popSize; i++)
    {
        new_pop[i] = genann_init(inputNeuronCnt, hiddenLayerCnt, hiddenNeuronCnt, outputNeuronCnt);
        int pA = idx[rand() % top];
        int pB = idx[rand() % top];

        for(int k = 0; k < weightCnt; k++)
        {
            float pAfitness = fitness[pA];
            float pBfitness = fitness[pB];
            double pAweight = population[pA]->weight[k];
            double pBweight = population[pB]->weight[k];
            
            double newWeight = (pAfitness * pAweight + pBfitness * pBweight) / ((double)pAfitness + (double)pBfitness);
            new_pop[i]->weight[k] = newWeight;
        }

        for (int j = 0; j < weightCnt; j++) 
        {
            double mutation = (double)RandomFloat(-GA.mutationLimit, GA.mutationLimit);
            if ((float)rand() / RAND_MAX < GA.mutationChance && new_pop[i]->weight[j] + mutation > -0.5f && new_pop[i]->weight[j] + mutation < 0.5)
                new_pop[i]->weight[j] += mutation;
        }
    }

    for (int i = 0; i < popSize; i++) {
        genann_free(population[i]);
        population[i] = genann_copy(new_pop[i]);
        genann_free(new_pop[i]);
    }
}


float smallest(int from, int to, float opponents[]) 
{
    float small = opponents[from];
    for (int i = from + 1; i <= to; i++) {
        if (small > opponents[i])
            small = opponents[i];
    }
    return small;
}


float* opponentProximity(float opponents[]) 
{
    closest = smallest(17, 18, opponents);
}


#define S 5
float med(float a[])
{
    float b[S];
    for (int i = 0; i < S; i++)
    {
        b[i] = a[i];
    }
    for (int i = 0; i < S - 1; i++)
    {
        for (int j = i + 1; j < S; j++)
        {
            if (b[j] < b[i])
            {
                float temp = b[i];
                b[i] = b[j];
                b[j] = temp;
            }
        }
    }

    return a[S / 2];
}

void shift(float* a, float newE)
{
    for (int i = 1; i < S; i++)
        a[i - 1] = a[i];

    a[S - 1] = newE;
}

#define REVERSE

structCarControl CDrive(structCarState cs)
{
#ifdef REVERSE
    //printf("\nREVERSE");
#endif
        
    shift(sensorHistory1, cs.track[1]);
    shift(sensorHistory5, cs.track[5]);
    shift(sensorHistory9, cs.track[9]);
    shift(sensorHistory13, cs.track[13]);
    shift(sensorHistory17, cs.track[17]);

    float track1 = med(sensorHistory1);
    float track5 = med(sensorHistory5);
    float track9 = med(sensorHistory9);
    float track13 = med(sensorHistory13);
    float track17 = med(sensorHistory17);

    if (cs.curLapTime < prevCurLapTime)
        lapsCompleted++;
    prevCurLapTime = cs.curLapTime;

    clutching(&cs, &clutch);
    int gear = getGear(&cs);
    int meta = 0;
    opponentProximity(cs.opponents);

    double input[inputNeuronCnt];
    bool filterOn = false;
    if (filterOn)
    {
        input[0] = (double)track1 / 5;
        input[1] = (double)track5 / 5;
        input[2] = (double)track9 / 5;
        input[3] = (double)track13 / 5;
        input[4] = (double)track17 / 5;
    }
    else
    {
        input[0] = (double)cs.track[1] / 5;
        input[1] = (double)cs.track[5] / 5;
        input[2] = (double)cs.track[9] / 5;
        input[3] = (double)cs.track[13] / 5;
        input[4] = (double)cs.track[17] / 5;
    }
    input[5] = (double)closest;
    input[6] = (double)cs.angle * 10; // c2s.angle [-3,14, +3,14] in radian
    input[7] = (double)cs.trackPos * 10;
    input[8] = (double)cs.speedX / 6;
    input[9] = (double)cs.speedY / 2;


    const double* prediction;
    if (mode == train_random || mode == train_continue)
        prediction = genann_run(population[GA.curIndividuum], input);
    else
        prediction = genann_run(inferenceNN, input);

    double accel = prediction[0];
    double brake = prediction[1];
    double steer = prediction[2] * 2 - 1;

    if (accel > brake)
        brake = 0;
    else
        accel = 0;

    if (cs.speedX > 149.0f)
        accel = 0.0f;

    if (mode == train_random || mode == train_continue)
    {
        evaluate(cs);
        if (cs.curLapTime > RewardPolicy.laptimeThd || stuck > maxStuck || lapsCompleted == GA.totalLaps - 1)
            meta = 1;
    }

    structCarControl cc = { accel, brake, gear, steer, clutch, meta };
    return cc;
}


void ConShutdown()
{
    printf("Bye bye!");
}


void LogGeneration()
{
    bestIdx = 0;
    for (int i = 0; i < popSize; i++) {
        if (fitness[i] > fitness[bestIdx])
            bestIdx = i;
    }
    printf("\nNEXT CYCLE:\t%2d", GA.curCycle);

    FILE* fp;
    fp = fopen(logPath, "a");
    char gen[200];
    sprintf(gen, "\n\n\nGeneration: %d\nBestIdx: %d\nFitness of bestIdx: %f\n", GA.curCycle, bestIdx, fitness[bestIdx]);
    fputs(gen, fp);
    fputs("\nfitness values", fp);

    float meanFit = 0;
    for (int i = 0; i < popSize; i++)
    {
        meanFit += fitness[i];
        char data[200];
        sprintf(data, "\n%d:\tfitness:\t%f", i, fitness[i]);
        fputs(data, fp);
    }
    
    char data[200];
    meanFit /= popSize;
    sprintf(data, "\nMean fitness:\t%f", meanFit);
    fputs(data, fp);
    fclose(fp);
}


void SaveGeneration()
{
    char dir[100];
    struct stat st = { 0 };
    sprintf(dir, "gen%03d", GA.curCycle);
    if (stat(dir, &st) == -1) {
        mkdir(dir, 0777);
    }

    for (int i = 0; i < popSize; i++)
    {
        char path[100];
        sprintf(path, "gen%03d/%02d.txt", GA.curCycle, i);
        FILE* out = fopen(path, "w");
        genann_write(population[i], out);
        fclose(out);
    }
}

void next() // next individuum or generation
{
    GA.curTry = 0;
    
    float sum = 0.0f;
    for(int i = 0; i < totalTries; i++)
        sum += fitnessPerTry[i];
    fitness[GA.curIndividuum] = sum / (float)totalTries;

    if (GA.curIndividuum == popSize - 1)
    {
        LogGeneration();
        SaveGeneration();        

        if (GA.curCycle == GA.cycles - 1)
            exit(420);
        
        reproduce();
        GA.curIndividuum = 0;
        GA.curCycle++;

        for (int i = 0; i < popSize; i++)
            fitness[i] = 1;
        for(int i = 0; i < totalTries; i++)
            fitnessPerTry[i] = 1;
        
        return;
    }

    for(int i = 0; i < totalTries; i++)
        fitnessPerTry[i] = 1;
    GA.curIndividuum++;
    fitness[GA.curIndividuum] = 1;
}

void ConRestart()
{
    stuck = 0;
    prevDistRaced = 0.0f;
    prevDamage = 0.0f;
    lapsCompleted = 0;
    prevCurLapTime = -10.0f;

    GA.curTry++;
    if(GA.curTry == totalTries)
        next();        
    
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