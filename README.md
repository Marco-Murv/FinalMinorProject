
# Basin Hopping

## Requirements
- ase >= ?.?.?
- NumPy >= 1.21
- PyYaml >= 5.1

## Usage
### Serial
`python basin_hopping.py -f ./config/config.yaml -v`
### Parallel
`mpirun -n 4 python basin_hopping.py -f config/config.yaml -v`

## Examples
|          Arguments         | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| `-f config/config.yaml`    | Run the basin hopping algorithm with settings defined in `config.yaml` |
| `-f config/config.json`    | Run the basin hopping algorithm with settings defined in `config.json` |
| `-f config/config.yaml -v` | Turn on log messages                                                   |
| `-n 10`                    | Run the basin hopping algorithm for a cluster with 10 atoms            |
| `-n 10 -m 500`             | Run the basin hopping algorithm for 500 steps                          |

## Full usage
```
usage: basin_hopping.py (-f CONFIG | -n CLUSTER_SIZE) [options]

Run the basin hopping algorithm.

required arguments:
  One of these arguments must be present

  -f CONFIG, --config CONFIG
                        The location of the config file
  -n CLUSTER_SIZE, --cluster-size CLUSTER_SIZE
                        The size of the cluster

optional arguments:
  -r RADIUS, --radius RADIUS
                        Radius of the sphere the initial atoms configuration
                        is uniformly distributed in
  -c MAX_RADIUS, --max-radius MAX_RADIUS
                        The maximum radius of the cube to constrain the atoms
                        in. If not set, no constraint is placed on the atoms
  --temperature TEMPERATURE
                        The temperature parameter for the Metropolis
                        acceptance criterion
  --step-size STEP_SIZE
                        The initial value of the step size
  --accept-rate ACCEPT_RATE
                        The desired step acceptance rate
  --step-size-factor STEP_SIZE_FACTOR
                        The factor to multiply and divide the step size by
  --step-size-interval STEP_SIZE_INTERVAL
                        The interval for how often to update the step size
  -m MAX_STEPS, --max-steps MAX_STEPS
                        The maximum number of steps the algorithm will take
  -ss STOP_STEPS, --stop-steps STOP_STEPS
                        The number of steps, without there being a new
                        minimum, the algorithm will take before stopping. If
                        not set, the algorithm will run for the maximum number
                        of steps
  -st STOP_TIME, --stop-time STOP_TIME
                        The maximum amount of time, in seconds, the algorithm
                        will run for before stopping. If not set, the
                        algorithm will run for the maximum number of steps

filter arguments:
  -fn, --no-filter      Use no filter
  -fs, --filter-significant-figures
                        Use significant figures filter
  -fd, --filter-difference
                        Use difference filter
  -sf SIGNIFICANT_FIGURES, --significant-figures SIGNIFICANT_FIGURES
                        Significant figures to round the potential energy to
                        to check for uniqueness
  -d DIFFERENCE, --difference DIFFERENCE
                        Minimum potential energy difference between unique
                        local minima to check for uniqueness

logging & help:
  -h, --help            show this help message and exit
  -v, --verbose         Print information about each step
  -db DATABASE, --database DATABASE
                        Database file for storing global minima
  -tr TRAJECTORY, --trajectory TRAJECTORY
                        Trajectory file for storing local minima
  -trf FILTERED_TRAJECTORY, --filtered-trajectory FILTERED_TRAJECTORY
                        Trajectory file for storing filtered local minima. If
                        None, filtered local minima are stored in the original
                        trajectory file
```