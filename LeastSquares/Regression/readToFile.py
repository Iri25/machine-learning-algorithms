import csv


def loadData(fileName, inputVariableName, outputVariableName):
    data = []
    dataNames = []

    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1

    selectedVariable = dataNames.index(inputVariableName)
    inputs = [float(data[i][selectedVariable]) for i in range(len(data))]

    selectedOutput = dataNames.index(outputVariableName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs, outputs


def loadData2(fileName, inputCols, outputCols):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1

    inputs = []

    for i in range(len(data)):
        line = []
        for column in inputCols:
            columnIndex = dataNames.index(column)
            line.append(float(data[i][columnIndex]))
        inputs.append(line)

    columnIndex = dataNames.index(outputCols)
    outputs = [float(data[i][columnIndex]) for i in range(len(data))]
    return inputs, outputs
