import csv

with open('senti') as f:
    reader = csv.reader(f,delimiter = '\t')
    new_dict = {}
    for row in reader:
        if row[0].startswith('#'):
            continue
        terms = row[4]
        first_term = terms.split('#')[0]
        #print(first_term)
        try:
            if float(row[2])==0 and float(row[3]) == 0.0:
                continue
        except ValueError:
            pass
        new_dict[first_term]= [float(row[2]), float(row[3])]


