#!/bin/bash

potentialFoam -writePhi

decomposePar -force
mpirun -np 4 vivPimpleFoam -parallel > log
reconstructPar -newTimes
