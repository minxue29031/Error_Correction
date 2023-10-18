import csv

class DataExtractor:
    def __init__(self, input_csv_file, output_csv_file, fail_fragments_file, succ_fragments_file, csv_encoding='utf-8'):
        self.input_csv_file = input_csv_file
        self.output_csv_file = output_csv_file
        self.fail_fragments_file = fail_fragments_file
        self.succ_fragments_file = succ_fragments_file
        self.csv_encoding = csv_encoding

    def extract_data(self):
        column_1 = []
        column_2 = []

        with open(self.input_csv_file, 'r', encoding=self.csv_encoding) as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row

            for row in reader:
                if len(row) >= 2:
                    column_1_tokens = row[0].split()
                    column_2_tokens = row[1].split()

                    if column_1_tokens:  # Check if the first column is not empty
                        if len(column_2_tokens) <= 4:
                            column_1.append(column_1_tokens[0:4])
                            column_2.append(column_2_tokens)
                        else:
                            column_2_last_four_to_last_one = column_2_tokens[-4:]  # Last four tokens of column 2
                            column_2_len = len(column_2_tokens)
                            print("column_2_len:", column_2_len)

                            column_1_start_pos = max(column_2_len - 4, 0)
                            column_1_end_pos = min(column_2_len + 5, len(column_1_tokens))
                            print("column_1_start_pos:", column_1_start_pos)
                            print("column_1_end_pos:", column_1_end_pos)                            
                            column_1_tokens_extracted = column_1_tokens[column_1_start_pos:column_1_end_pos]  # Tokens from column 1: POSi-2 to POSj+2

                            column_1.append(column_1_tokens_extracted)
                            column_2.append(column_2_last_four_to_last_one)

        return column_1, column_2


    def write_to_csv(self, column_1, column_2):
        unique_rows = set()
        with open(self.output_csv_file, 'w', newline='', encoding=self.csv_encoding) as file:
            writer = csv.writer(file)
            writer.writerow(['fail_fragment', 'succ_fragment'])

            for i in range(len(column_1)):
                row = (' '.join(column_1[i]), ' '.join(column_2[i]))
                if row not in unique_rows:
                    writer.writerow(row)
                    unique_rows.add(row)


    def write_to_file(self, data, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            for sentence in data:
                file.write(' '.join(sentence) + '\n')


    def process_data(self):
        column_1, column_2 = self.extract_data()

        # Remove empty lines from normal_fail_fragments.tok and delete the corresponding indexed lines
        filtered_column_1 = []
        filtered_column_2 = []
        for i, sentence in enumerate(column_1):
            if sentence:
                filtered_column_1.append(sentence)
                filtered_column_2.append(column_2[i])

        self.write_to_csv(filtered_column_1, filtered_column_2)

        # Get deduplicated results from write_to_csv
        deduplicated_column_1 = []
        deduplicated_column_2 = []
        with open(self.output_csv_file, 'r', encoding=self.csv_encoding) as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                deduplicated_column_1.append(row[0].split(' '))
                deduplicated_column_2.append(row[1].split(' '))

        fail_fragments = [sentence_tokens for sentence_tokens in deduplicated_column_1]
        succ_fragments = [sentence_tokens for sentence_tokens in deduplicated_column_2]

        self.write_to_file(fail_fragments, self.fail_fragments_file)
        self.write_to_file(succ_fragments, self.succ_fragments_file)

        print("Extracted data for column 1 (POSi-2 to POSj+2):")
        for i in range(len(deduplicated_column_1)):
            tokens = deduplicated_column_1[i]
            print(f"Sentence {i+1}: {' '.join(tokens)}")

        print("Extracted data for column 2 (last four tokens):")
        for i in range(len(deduplicated_column_2)):
            tokens = deduplicated_column_2[i]
            print(f"Sentence {i+1}: {' '.join(tokens)}")

# Usage
# extractor_normal = DataExtractor('extract_normal_pairs.csv', 'normal_succ_fail_fragment_pairs.csv', 'normal_fail_fragments.tok', 'normal_succ_fragments.tok', 'gbk')
# extractor_normal.process_data()

# extractor_EOS = DataExtractor('extract_EOS_pairs.csv', 'EOS_succ_fail_fragment_pairs.csv', 'EOS_fail_fragments.tok', 'EOS_succ_fragments.tok', 'gbk')
# extractor_EOS.process_data()


extractor_EOS = DataExtractor('extract_AAA.csv', 'EOS_succ_fail_fragment_pairs.csv', 'EOS_fail_fragments.tok', 'EOS_succ_fragments.tok', 'gbk')
extractor_EOS.process_data()
