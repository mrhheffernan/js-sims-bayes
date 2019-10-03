import sys
import pandas as pd

#number of events that each design point should have
target_nev = int(sys.argv[1])

#a tolerance
tolerance = int(sys.argv[2])

all_events = []

#read in the files
for idf in [0,1,2,3]:
    filename = "event_summaries_Pb_Pb_2760/event_summaries_main/event_summary_idf_" + str(idf) + ".dat"
    summ = pd.read_csv(filename, sep=' ')
    nev = summ['nev']

    problem_points = nev.loc[ nev < (target_nev - tolerance) ]
    problem_indices = problem_points.index.values

    all_events.append(problem_indices)

    print("Problematic design points for idf 0 : ")
    print(problem_points)

set0 = set(all_events[0])
set1 = set(all_events[1])
set2 = set(all_events[2])
set3 = set(all_events[3])

common_set = set3.intersection(set2.intersection(set1.intersection(set0)))
print("Problematic design points common to all delta f : ")
print(common_set)
