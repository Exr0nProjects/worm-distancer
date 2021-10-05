import pandas

FILENAME = "worm_distances.tsv"

if __name__ == '__main__':
    data = pandas.read_csv(FILENAME, sep="	")
    data = data.set_index('time')

    info = data.iloc[:,0:1]
    skeleton = data.iloc[:,1:]

    skeleton = skeleton.diff();

    for c1, c2 in zip(skeleton.iloc[:,:-1].itercols(),skeleton.iloc[:,1:].itercols()):
        print(c1, c2)


