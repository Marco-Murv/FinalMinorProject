# Genetic Algorithm

Genetic algorithm for global geometry optimisation

This folder contains three algorithms for the gentic algorithm approach.
`genetic_algorithm.py` is a sequential version, whereas `ga_distributed.py` 
and `ga_sub_populations.py` are parallel implementations using `mpi4py`.

## Configuration
The directory `genetic_algorithm/config/` contains a config file for each of
the algorithms. All of the parameters there can also be changed from the
command line. Command line has priority. Use the `-h` option to get a help menu.

## How to run
### Run `genetic_algorithm.py`

```
python3 genetic_algorithm.py
```

### Run `ga_distributed.py`

```
mpiexec -n P python3 ga_distributed.py 
```
* `P` is the number of processors
* Specify configuration in config file or by terminal arguments
* Add `-h` for help menu

### Run `ga_sub_population.py`

```
mpiexec -n P python3 ga_sub_populations.py 
```
* Specify configuration in config file or by terminal arguments
* Add `-h` for help menu