#!/usr/bin/python3

import ase.db
from ase.visualize import view
import sys
import os


def show_db(db):
    print("\nShowing the contents of the database:")
    print("-----------------------------------------------------------------------")
    print("| ID | Cluster Size | Pop. Size | Max Gens | Max NO Success |  Energy |")
    print("-----------------------------------------------------------------------")
    for i in range(len(db)):
        row = db[i+1]
        print(f"|{row.id:3d} |{row.cluster_size:13d} |{row.pop_size:10d} |"
              f"{row.max_gens:9d} |{row.max_no_success:15d} |{row.energy:8.3f} |")
    print("----------------------------------------------------------------------")


def show_help():
    print("\nProvide one or more valid ID from the database to view.\n"
          "Provide a negative number to show the database\n"
          "Write 'h' to show this help\n"
          "Write 'q' or 'Q' to quit")


db_file_relative = "genetic_algorithm_results.db"

dirname = os.path.dirname(__file__)
db_file = os.path.join(dirname, db_file_relative)


while not os.path.exists(db_file):
    db_file = input(f"{db_file} is not an existing database. \n\n\tNew name: ")


db = ase.db.connect(db_file)

view_ids = ['0']

if len(sys.argv) > 1:
    view_ids = sys.argv[1:]
else:
    show_help()


while True:
    while any([int(view_id) <= 0 or int(view_id) > len(db) for view_id in view_ids]):
        if not str(0) in view_ids:
            show_db(db)

        # view_id = list(input(f"\n\tView ID: "))
        view_ids = list(input("\n\tView ID(s) : ").strip().split())

        try:
            view_ids = [int(view_id) for view_id in view_ids]

        except ValueError:
            if 'q' in view_ids or 'Q' in view_ids:
                sys.exit("\n\nExiting...\n")

            else:
                print("ERROR: Invalid option.")
                show_help()
                view_ids = ['0']

    for view_id in view_ids:
        print(f"Viewing the cluster with ID {view_id}.")
        view(db[int(view_id)].toatoms())

    print()
    view_ids = [str(0)]
