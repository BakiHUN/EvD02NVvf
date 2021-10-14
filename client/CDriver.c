#include "CDriver.h"
#include "genann.h"

/* Gear Changing Constants*/
const int gearUp[6]=
    {
        5000,6000,6000,6500,7000,0
    };
const int gearDown[6]=
    {
        0,2500,3000,3000,3500,3500
    };

/* Stuck constants*/
const int stuckTime = 25;
const float stuckAngle = .523598775; //PI/6

/* Accel and Brake Constants*/
const float maxSpeedDist=70;
const float maxSpeed=150;
const float sin5 = 0.08716;
const float cos5 = 0.99619;

/* Steering constants*/
const float steerLock=0.785398;
const float steerSensitivityOffset=80.0;
const float wheelSensitivityCoeff=1;

/* ABS Filter Constants */
const float wheelRadius[4]={0.3179,0.3179,0.3276,0.3276};
const float absSlip=2.0;
const float absRange=3.0;
const float absMinSpeed=3.0;

/* Clutch constants */
const float clutchMax=0.5;
const float clutchDelta=0.05;
const float clutchRange=0.82;
const float clutchDeltaTime=0.02;
const float clutchDeltaRaced=10;
const float clutchDec=0.01;
const float clutchMaxModifier=1.3;
const float clutchMaxTime=1.5;

int stuck;
float clutch;


// CUSTOM STUFF FROM HERE
float prevDistRaced;
float laptimeThd = 5.0f;

int cycles = 50;
float mutationChance = 0.05f;

#define popSize 30 
genann* population[popSize];
int fitness[popSize];

int currentIndividual = 0;
int currentCycle = 0;


// neural network architecture
#define inputNeuronCnt 22
#define hiddenLayerCnt 2
#define hiddenNeuronCnt 13
#define outputNeuronCnt 3






// only for program flow exploration
int counter = 0;


//gives 19 angles for the distance sensors
void Cinit(float* angles)
{
    for (int i = 0; i < popSize; i++)
        population[i] = genann_init(inputNeuronCnt, hiddenLayerCnt, hiddenNeuronCnt, outputNeuronCnt);
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
    int points = 0;

    // going is good mmmmkkaaaayyy???
    points += (int)cs.distRaced - prevDistRaced;
    //printf("\n%f", cs.distRaced - prevDistRaced);
    prevDistRaced = cs.distRaced;

    if (fitness[currentIndividual] + points < 1)
        fitness[currentIndividual] = 1;
    else
        fitness[currentIndividual] = points;
}


void next()
{
    if (currentCycle == cycles - 1)
    {
        //stop and save the best individual to a file
        return;
    }

    if (currentIndividual == popSize - 1)
    {
        // create pool for next generation
        // crossover with mutation
        currentIndividual = 0;
        currentCycle++;
        return;
    }

    currentIndividual++;
}

/*
    todo:
    - maybe use same magnitude between input data
        e. g. distance sensor gives back 200
        divide by 10?
        reason to do so: after a certain speed
        the network becomes blind to any change
        ha 45 felett minden 1 lesz a sigmoid után
        akkor nem fog értesulni arrol hogy 100 megy vagy 50 nel
        
*/

//double const* feed_forward(structCarState cs, float input[], )

structCarControl CDrive(structCarState cs)
{

    clutching(&cs, &clutch);
    int gear = getGear(&cs); 

    // possible addition to input:
    // z, trackpos
    float input[inputNeuronCnt];
    for (int i = 0; i < 19; i++)
        input[i] = cs.track[i];
    input[19] = cs.speedX;
    input[20] = cs.speedY;
    input[21] = cs.speedZ;

    // feed forward
    double const* prediction = genann_run(population[currentIndividual], input);
    double accel = prediction[0];
    double brake = prediction[1];
    double steer = prediction[2] * 2 - 1;
    
    if (accel > brake)
        brake = 0;
    else
        accel = 0;

    evaluate(cs);

    if (0)
    {
        printf("\n\n\ninput data:");
        for (int i = 0; i < inputNeuronCnt; i++)
            printf("\n%d:\t%f", i, input[i]);

        printf("\n\n\nprediction:");
        printf("\naccel:\t%f", accel);
        printf("\nbrake:\t%f", brake);
        printf("\nsteer:\t%f", steer);
        printf("\nfitness:\t%d", fitness[currentIndividual]);

        printf("\n\nspeedx:\t%f", cs.speedX);
        printf("\nspeedy:\t%f", cs.speedY);
        printf("\nspeedz:\t%f", cs.speedZ);
    } 

    int meta = 0;
    if (cs.curLapTime > laptimeThd)
    {
        meta = 1;
        next();
    }

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

    printf("\n\ncounter:\t%d\n\n", counter);
    counter += 1;
    printf("Restarting the race!");
}


int getGear(structCarState *cs)
{

    int gear = cs->gear;
    int rpm  = cs->rpm;

    // if gear is 0 (N) or -1 (R) just return 1 
    if (gear<1)
        return 1;
    // check if the RPM value of car is greater than the one suggested 
    // to shift up the gear from the current one     
    if (gear <6 && rpm >= gearUp[gear-1])
        return gear + 1;
    else
    	// check if the RPM value of car is lower than the one suggested 
    	// to shift down the gear from the current one
        if (gear > 1 && rpm <= gearDown[gear-1])
            return gear - 1;
        else // otherwhise keep current gear
            return gear;
}

float getSteer(structCarState *cs)
{
	// steering angle is compute by correcting the actual car angle w.r.t. to track 
	// axis [cs->angle] and to adjust car position w.r.t to middle of track [cs->trackPos*0.5]
    float targetAngle=(cs->angle-cs->trackPos*0.5);
    // at high speed reduce the steering command to avoid loosing the control
    if (cs->speedX > steerSensitivityOffset)
        return targetAngle/(steerLock*(cs->speedX-steerSensitivityOffset)*wheelSensitivityCoeff);
    else
        return (targetAngle)/steerLock;

}

float getAccel(structCarState *cs)
{
    // checks if car is out of track
    if (cs->trackPos < 1 && cs->trackPos > -1)
    {
        // reading of sensor at +5 degree w.r.t. car axis
        float rxSensor=cs->track[10];
        // reading of sensor parallel to car axis
        float cSensor=cs->track[9];
        // reading of sensor at -5 degree w.r.t. car axis
        float sxSensor=cs->track[8];

        float targetSpeed;

        // track is straight and enough far from a turn so goes to max speed
        if (cSensor>maxSpeedDist || (cSensor>=rxSensor && cSensor >= sxSensor))
            targetSpeed = maxSpeed;
        else
        {
            // approaching a turn on right
            if(rxSensor>sxSensor)
            {
                // computing approximately the "angle" of turn
                float h = cSensor*sin5;
                float b = rxSensor - cSensor*cos5;
                float sinAngle = b*b/(h*h+b*b);
                // estimate the target speed depending on turn and on how close it is
                targetSpeed = maxSpeed*(cSensor*sinAngle/maxSpeedDist);
            }
            // approaching a turn on left
            else
            {
                // computing approximately the "angle" of turn
                float h = cSensor*sin5;
                float b = sxSensor - cSensor*cos5;
                float sinAngle = b*b/(h*h+b*b);
                // estimate the target speed depending on turn and on how close it is
                targetSpeed = maxSpeed*(cSensor*sinAngle/maxSpeedDist);
            }

        }

        // accel/brake command is expontially scaled w.r.t. the difference between target speed and current one
        return 2/(1+exp(cs->speedX - targetSpeed)) - 1;
    }
    else
        return 0.3; // when out of track returns a moderate acceleration command

}





float filterABS(structCarState *cs,float brake)
{
	// convert speed to m/s
	float speed = cs->speedX / 3.6;
	// when spedd lower than min speed for abs do nothing
    if (speed < absMinSpeed)
        return brake;
    
    // compute the speed of wheels in m/s
    float slip = 0.0f;
    for (int i = 0; i < 4; i++)
    {
        slip += cs->wheelSpinVel[i] * wheelRadius[i];
    }
    // slip is the difference between actual speed of car and average speed of wheels
    slip = speed - slip/4.0f;
    // when slip too high applu ABS
    if (slip > absSlip)
    {
        brake = brake - (slip - absSlip)/absRange;
    }
    
    // check brake is not negative, otherwise set it to zero
    if (brake<0)
    	return 0;
    else
    	return brake;
}



void clutching(structCarState *cs, float *clutch)
{
  float maxClutch = clutchMax;

  // Check if the current situation is the race start
  if (cs->curLapTime<clutchDeltaTime  && cs->stage == RACE && cs->distRaced < clutchDeltaRaced)
    *clutch = maxClutch;

  // Adjust the current value of the clutch
  if(clutch > 0)
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
	*clutch = fmin(maxClutch,*clutch);
    
	// if clutch is not at max value decrease it quite quickly
	if (*clutch!=maxClutch)
	{
	  *clutch -= delta;
	  *clutch = fmax(0.0,*clutch);
	}
	// if clutch is at max value decrease it very slowly
	else
		*clutch -= clutchDec;
  }
}



/*
    if(cs.stage != cs.prevStage)
    {
        cs.prevStage = cs.stage;
    }
    // check if car is currently stuck
    if ( fabs(cs.angle) > stuckAngle )
    {
        // update stuck counter
        stuck++;
    }
    else
    {
        // if not stuck reset stuck counter
        stuck = 0;
    }

    // after car is stuck for a while apply recovering policy
    if (stuck > stuckTime)
    {
        //set gear and sterring command assuming car is
        // pointing in a direction out of track

        // to bring car parallel to track axis
        float steer = - cs.angle / steerLock;
        int gear=-1; // gear R

        // if car is pointing in the correct direction revert gear and steer
        if (cs.angle*cs.trackPos>0)
        {
            gear = 1;
            steer = -steer;
        }

        // Calculate clutching
        clutching(&cs,&clutch);

        // build a CarControl variable and return it
        structCarControl cc = {1.0f,0.0f,gear,steer,clutch};
        return cc;
    }

    else // car is not stuck
    {
        // compute accel/brake command
        float accel_and_brake = getAccel(&cs);
        // compute gear
        int gear = getGear(&cs);
        // compute steering
        float steer = getSteer(&cs);


        // normalize steering
        if (steer < -1)
            steer = -1;
        if (steer > 1)
            steer = 1;

        // set accel and brake from the joint accel/brake command
        float accel,brake;
        if (accel_and_brake>0)
        {
            accel = accel_and_brake;
            brake = 0;
        }
        else
        {
            accel = 0;
            // apply ABS to brake
            brake = filterABS(&cs,-accel_and_brake);
        }

        // Calculate clutching
        clutching(&cs,&clutch);

        // build a CarControl variable and return it
        structCarControl cc = {accel,brake,gear,steer,clutch};
        return cc;
    }*/