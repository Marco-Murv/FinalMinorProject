# Artificial Bee Colony Algorithm

Artificial bee Colony algorithm for global geometry optimisation

`artificial_bee_colony_algorithm.py` runs in parallel using 'mpi4py'.

### Employeed Bee
### Onlooker Bee
### Scout Bee

## Configuration
The directory `/config` contains a config file for each of
the algorithms


## How to run
### Run `artificial_bee_colony_algorithm.py`
```
mpiexec -n P python3 artificial_bee_colony_algorithm.py 
```
### Configuration

```yaml
cluster_config:
  # Radius of the sphere the initial atoms configuration
  # is uniformly distributed in   
  cluster_radius: 2.5
  # the size of the cluster
  cluster_size: 20
  # number of clusters in the population
  pop_size: 24
employed_bee_config:
  # Enable the employed bee
  enable: 1
  # monte carlo search is x_i +f*(x_k1+x_k2-x_k3-x_k3)
  # where k1 ~ k4 is random cluster from the population
  monte_carlo_search_f: 0.035
  # there is two search method which is mutation search (0)
  # and monte carlo search (1)
  search_method: 1
  # for monte carlo search size is total number of clusters positions
  # subtracted and added and for mutation search 3 search size means
  # trigonometric search 
  search_size: 4
onlooker_bee_config:
  enable: 1
run_config:
  # eg potential energy is compared with current loop and index of loop devided by the auto_stop value
  # if no meaningful improvement or change is discoverd the loop will stop. To disable -1.
  auto_stop: 1.4
  # significant values considered when two potential energy is compared
  auto_stop_sf: 2
  # algorithm can run in parallel (1) or only on single processor (0)
  is_parallel: 1
  # maximum cycle ignoring other factors such as timeout, or auto stop
  maximum_cycle: 999999999
  # minimum cycle ignoring other factors such as timeout, or auto stop
  minimum_cycle: 10
  # run id, increments on each loop
  run_id: 19
  # Stopping the loop after set time. To disable -1.
  time_out: 30
  # view trajectory at the end
  view_traj: 0
scout_bee_config:
  # the scout bee removes a cluster when a cluster is not updated for x loops on other bees and replaces 
  # it with a new cluster to look for other local minima
  check_energies_every_x_loops: 6
  # enable/disable cluster renewal when cluster is not updated for x loops
  update_energies: 1
  # specifies after how many iterations an unchanged energy from a cluster should be considered a local minima
  count: 3
  # enable/disable the scout bee
  enable: 1
  # scout bee scans through the list of current clusters and finds the cluster with the lowest energy. It only
  # continues with the clusters that are close to the lowest energy found. How close the energies should be is specified 
  # by this value. If the lowest energy is -10 and energy_abnormal = 0.6, then it only continues with clusters that have 
  # energies between -10 - (-10 * 0.6) = -4 and -10.
  energy_abnormal: 0.6
  # if two clusters are (almost) similar in energies, it only keeps the lowest. Similarity is based on this value. For
  # example: there are two clusters with energies -74 and -75 and this value is set to 1, it will only keep -75.
  energy_difference: 0.06
```
