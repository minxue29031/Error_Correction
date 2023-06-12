import csv
import re
import pandas as pd

def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()

    sentences = content.split('\n')

    processed_lines = []
    for sentence in sentences:
        words = sentence.split()
        processed_lines.extend(words)
        processed_lines.append('')

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["target_token"])
        writer.writerows([[word] for word in processed_lines])


def pre_extract_normal_pairs(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()

    content = content.replace("</s>", "")

    pattern = r"\[\[array\(\[(.*?)\],"
    matches = re.findall(pattern, content, re.S)

    data = []
    for match in matches:
        parts = re.split(r'",|\',|,",', match)
        if len(parts) >= 1:
            first_column = re.sub(r"^[\'\",\s]+|[\'\",\s]+$", '', parts[0])
            second_column = ""
            if len(parts) >= 2:
                second_column = re.sub(r"^[\'\",\s]+|[\'\",\s]+$", '', parts[1])
            data.append([first_column, second_column])

    with open(output_file, "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["fail_input", "normal_prefix"])
        writer.writerows(data)


def generate_extract_normal_pairs(output_token_file, extract_normal_pairs_file, updated_extract_normal_pairs_file):
    # Read the output_token.csv file
    output_df = pd.read_csv(output_token_file)

    # Read the extract_normal_pairs.csv file
    extract_df = pd.read_csv(extract_normal_pairs_file)
    
    # Drop rows from the second column where the corresponding cell in the first column is empty
    extract_df = extract_df.dropna(subset=['fail_input'])

    # Iterate over each row in extract_normal_pairs.csv
    for index, row in extract_df.iterrows():
        if pd.isnull(row['normal_prefix']):
            extract_df.at[index, 'normal_prefix'] = str(output_df.at[index, 'target_token'])
        else:
            target_token = output_df.at[index, 'target_token']
            if pd.isnull(target_token):
                target_token = ''
            extract_df.at[index, 'normal_prefix'] += ' ' + str(target_token)

    # Save the updated data to a new CSV file
    extract_df.to_csv(updated_extract_normal_pairs_file, index=False)


def main():
    # Call the function to process the file and write the results to "output_token.csv"
    process_file('output_80100_1_2_3.sa.tok', 'output_token.csv')

    # Call the function to generate extract_normal_pairs.csv
    pre_extract_normal_pairs('search_inputs.sa.tok', 'pre_extract_normal_pairs.csv')

    # Call the function to update extract_normal_pairs.csv
    generate_extract_normal_pairs('output_token.csv', 'pre_extract_normal_pairs.csv', 'extract_normal_pairs.csv')


if __name__ == '__main__':
    main()
