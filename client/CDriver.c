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
4.93774224066896527674e-01,
-3.79146702475051111669e-01,
-2.05847346415601317737e-02,
3.90743736075930048202e-01,
7.63115329447310308808e-02,
-3.80092776268806997741e-01,
-3.88607440412610249947e-01,
-7.62962736899908122723e-05,
-4.44547868282113123328e-01,
-2.04641865291299185348e-01,
-3.53297524948881502826e-01,
7.79747917111728305883e-03,
-2.06198309274575053429e-01,
4.83764152958769511059e-01,
-1.97225867488631856350e-01,
-4.19034394360179418548e-01,
-1.65059358500930841718e-01,
4.24039429914243037878e-01,
-2.46116519669179356633e-01,
-4.76836451307718123260e-01,
2.87774285103915561379e-01,
2.29300820947904937519e-01,
-1.59108249153111336316e-01,
-2.28690450758384955510e-01,
-3.70036927396465986639e-02,
1.02771080660420510711e-01,
4.83260597552415127609e-02,
8.91903439436018130593e-02,
-8.24457533494064032098e-02,
3.29218420972319680828e-01,
-3.44630268257698268997e-01,
-2.60795922727134033003e-01,
-4.36277352214117841722e-01,
-4.27335428937650696479e-01,
4.67040009765923080920e-01,
4.14456617938779858790e-01,
-1.29703665272989931978e-03,
-4.87365337076937188421e-01,
3.13882869960631083472e-02,
2.11142307809686569087e-01,
2.11478011413922528661e-01,
-1.51173436689352069795e-01,
-2.50267036957914967843e-01,
-7.60063478499710121206e-02,
3.22473830378124382001e-01,
2.06686605426190994628e-01,
2.85424359874263733339e-01,
4.44547868282113123328e-01,
-4.86999114963225177011e-01,
-2.56813257240516290558e-02,
1.77510910367137619836e-01,
-4.55504013183996081526e-01,
-1.99667348246711617854e-01,
-3.97427289651173465401e-01,
6.62404248176519105940e-02,
-1.74733726004821898758e-01,
3.14477980895413122830e-01,
3.99044770653401270621e-01,
4.35697500534073967060e-01,
1.14337595751823450918e-01,
2.33542893765068537704e-01,
3.47407452620014045586e-01,
2.80449842829676221356e-01,
9.05331583605456513553e-02,
3.33414716025269841637e-02,
-1.62816248054444834281e-02,
-3.97671437726981436001e-01,
4.18820764793847444274e-01,
2.07663197729422877025e-01,
-1.39790032654805163048e-01,
1.43421735282448836735e-01,
1.76259651478621770515e-01,
1.15619373149815407587e-01,
-3.71333964049195830448e-01,
2.61558885464033941126e-01,
-1.49800103762932235174e-01,
1.12109744560075719200e-01,
2.73674733726004815182e-01,
3.54976042970061356208e-01,
-3.10083315530869485510e-01,
-3.82259590441602847832e-01,
-4.72136600848414578202e-01,
-2.58354441969054215988e-01,
-3.06909390545365756697e-01,
2.84600360118411832566e-01,
1.98370311593976111908e-04,
4.00265511032441123618e-01,
8.22473830378123160756e-03,
-8.63216040528580474955e-02,
4.77767265846735789125e-02,
4.78514969328897921130e-01,
3.44401379436628518604e-02,
2.80816064943388177255e-01,
-7.63115329447309753697e-02,
-2.45521408734397650342e-02,
-3.81481368449954061361e-04,
-4.73113193151646460599e-01,
2.31162450025940713338e-01,
-4.96215704824976344689e-01,
2.98608355967894478766e-01,
2.68822290719321288499e-01,
2.15659047212134136196e-01,
-4.61424604022339535092e-01,
3.72066408276619742246e-01,
8.74507889034699115172e-02,
-7.38395336771752175409e-02,
5.83666493728446367228e-02,
4.07773064363536441590e-01,
-4.51353495895260470316e-01,
-2.23227637562181446818e-01,
-1.46946623126926745329e-02,
-3.54426709799493422359e-01,
4.14029358806115910241e-01,
-4.59929197058015715172e-01,
3.30347605822931655872e-01,
-4.33133945738090164745e-01,
1.96890163884395841265e-01,
3.03247169408246142197e-01,
4.49674977874080616935e-01,
-2.49870296334727015619e-01,
-2.79168065431684320199e-01,
4.18393505661183495725e-01,
3.18659016693624730365e-01,
-3.26136051515243996501e-01,
-1.64021729178746911160e-01,
-2.18054750205999847701e-02,
2.29880672627948867692e-01,
2.48008667256691128777e-01,
-5.00961333048494106279e-02,
-1.51539658803064081205e-01,
-1.78945280312509558129e-01,
3.54884487441633367233e-01,
-4.70915860469374669695e-01,
-2.11874752037110480885e-01,
-1.60817285683767186022e-01,
1.81600390636921238396e-01,
2.27805413983581006576e-01,
-3.11243018890957401368e-01,
-4.52391125217444400874e-01,
-3.75728629413739412257e-01,
-2.00582903530991563112e-01,
4.64323862422559296981e-01,
1.81539353617969245747e-01,
-4.53886532181768220795e-01,
4.47935422833948826415e-01,
-3.77559739982299247263e-01,
3.58882412182988996818e-01
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
#define inputNeuronCnt 14
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
enum Mode mode = inference;

genann* population[popSize];
genann* inferenceNN = NULL;
float fitness[popSize];
float fitnessPerTry[totalTries];
float closest[5];

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
    closest[0] = smallest(4, 11, opponents);
    closest[1] = smallest(12, 17, opponents);
    closest[2] = smallest(18, 23, opponents);
    closest[3] = smallest(24, 31, opponents);

    float rearRight = smallest(32, 35, opponents);
    float rearLeft = smallest(0, 3, opponents);
    closest[4] = (rearRight > rearLeft) ? rearLeft : rearRight;
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

structCarControl CDrive(structCarState cs)
{
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
    input[5] = (double)closest[0];
    input[6] = (double)closest[1];
    input[7] = (double)closest[2];
    input[8] = (double)closest[3];
    input[9] = (double)closest[4];
    input[10] = (double)cs.angle * 10; // c2s.angle [-3,14, +3,14] in radian
    input[11] = (double)cs.trackPos * 10;
    input[12] = (double)cs.speedX / 6;
    input[13] = (double)cs.speedY / 2;


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