import csv

input_filename = 'input_file.csv'
output_filename = 'output_file.csv'

with open(input_filename, 'r', encoding='gbk') as input_file, open(output_filename, 'w', newline='', encoding='gbk') as output_file:

    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    for row in reader:
        long_string = row[0][:-1]
        short_string = row[1]

        short_tokens = short_string.split(' ')

        if all(token in long_string for token in short_tokens):
            row.append(1)
        else:
            row.append(0)

        writer.writerow(row)

