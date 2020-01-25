import wave

#python wave library embeddings
def getEmbedding(filenames):
    #the output is a list of files
    output = []

    #find the shortest file length
    for file in filenames:
        a = wave.openfp(file, mode="rb")
        try:
            shortest
        except NameError:
            var_exists = False
        else:
            var_exists = True
        if not var_exists:
            shortest = a.getnframes()
        else:
            if (shortest > a.getnframes()):
                shortest = a.getnframes()

    #embed each file at the shortest length
    for file in filenames:
        a = wave.openfp(file, mode="rb")
        b = a.readframes(shortest)
        output.append(b)

    return output