
# Global optimisation algorithms

## Description

Atomic clusters have many applications in various areas, mainly because of their unique properties that depend on the geometrical structures of clusters. However, finding optimal stable structures is a computationally expensive task. Several global optimisation methods have been developed for this purpose, including [Basin Hopping](basin_hopping/), [Genetic Algorithm](genetic_algorithm/)s, and [Artificial Bee Colony algorithm](artificial_bee_colony_algorithm/)s. Parallel versions ofthese three were implemented and compared in the present work. The Basin Hopping algorithm converges very fast, but given more time and resources, the other two find global minima with lower energy. The Genetic Algorithm consistently finds the moststable configurations, but a precise setup is required to acquire the best result.

## Dependencies

Before trying to run any of the algorithms, make sure that all the required `Python3` modules are installed correctly. On Unix-like systems, it should be sufficient to run the following from the project home directory.

```bash
    python3 -m pip instal -r requirements.txt
```

## Usage

For the usage of each of the three algorithms, please refer to their dedicated `README.md` files.

* Artificial Bee Colonies: [artificial_bee_colony_algorithm/README.md](artificial_bee_colony_algorithm/README.md)
* Basin Hopping: [basin_hopping/README.md](basin_hopping/README.md)
* Genetic Algorithm: [genetic_algorithm/README.md](genetic_algorithm/README.md)
