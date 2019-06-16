import os

def isContainOnly(s):
    allowed_chars = {'\n'}
    s_set = (set(''.join(s)))
    return s_set.issubset(allowed_chars)

def cleanFolder(preCleanPath, postCleanPath, outPath):

    outPathfile = os.path.join(outPath, "out.txt")
    for preCleanFilePath in os.listdir(preCleanPath):

        preCleanFilename = os.path.join(preCleanPath, preCleanFilePath)
        postCleanFilename = os.path.join(postCleanPath, preCleanFilePath)
        postCleanFile = open(postCleanFilename, "w")
        print(f'Pre Clean File: {preCleanFilename} Post Clean File: {postCleanFilename}')

        #enchode file
        try:
            preCleanFile = open(preCleanFilename, encoding="utf8")
            lines = preCleanFile.readlines();

        except UnicodeError:
            out = open(outPathfile, "w")
            message = f'failed to enchod utf-8 {preCleanFilename}'
            out.write(message)
            print(message)
            out.close()
            continue;

        #enchode content

        for line in lines:
            words = line.split(" ")
            for word in words:
                if ((word.replace("_", "-").replace("\n", " ")).isprintable()) and (isContainOnly(word) != True):
                    try:
                     postCleanFile.write(word)
                    except:
                        continue;
                else:
                    print(word)

preCleanPath = "C:/Users/bb02/Desktop/data/Texts_Others"
postCleanPath = "C:/Users/bb02/Desktop/CleanDataUpdate/Texts_Others"
outPath = "C:/Users/bb02/Desktop/cleaning/"

cleanFolder(preCleanPath, postCleanPath, outPath)