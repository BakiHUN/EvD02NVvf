with open('ferenc.txt') as f:
    line = f.readline()
    items = line.split()

    with open('formatted_ferenc.txt', 'w') as writer:
        for i in items:
            writer.write(i + '\n')
