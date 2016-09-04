testFile = open("test.dat","r")
Sentiment_arr = []
Test_arr = []
for line in testFile:
    lineArr = line.split('\t',1)
    Sentiment_arr.append(lineArr[0])
    Test_arr.append(lineArr[1])

#for idx, line in enumerate(Test_arr):
#    print(Test_arr[idx])
#    print(Sentiment_arr[idx])


