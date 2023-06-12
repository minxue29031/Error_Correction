import re
import csv

def process_search_files(input_value_file, input_inputs_file, output_file):
    with open(input_value_file, 'r') as file:
        pre_search_value_content = file.readlines()

    with open(input_inputs_file, 'r') as file:
        pre_search_inputs_content = file.readlines()

    search_results = []

    for index, line in enumerate(pre_search_value_content):
        number = int(line.strip())
        if number == 1:
            if index < len(pre_search_inputs_content):
                search_results.append(pre_search_inputs_content[index])

    with open(output_file, 'w') as file:
        file.writelines(search_results)

    print("Processing complete! Results have been saved to the file", output_file, ".")

def process_search_inputs(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()

    content = content.replace('\n', '')
    content = re.sub(r'\[\[array\(', '\n[[array(', content)
    content = re.sub(r'^\n', '', content)

    with open(output_file, 'w') as file:
        file.write(content)

    print("Processing complete! Results have been saved to the file", output_file, ".")

def extract_values(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()

    results = re.findall(r'\[\[(.*?)\,', content)
    result_string = '\n'.join(results)

    with open(output_file, 'w') as file:
        file.write(result_string)

    print("Processing complete! Results have been saved to the file", output_file, ".")

def pre_extract_EOS_pairs(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()

    content = content.replace("</s>", "")

    pattern = r"\[\[array\(\[(.*?)\],"
    matches = re.findall(pattern, content, re.S)

    data = []
    for match in matches:
        parts = re.split(r'",|\',|,",', match)  # Splitting with Regular Expressions
        if len(parts) >= 2:
            first_column = re.sub(r"^[\'\",\s]+|[\'\",\s]+$", '', parts[0])
            second_column = re.sub(r"^[\'\",\s]+|[\'\",\s]+$", '', parts[1])
            data.append([first_column, second_column])



    with open(output_file, "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["fail_input", "succ_prefix"])
        writer.writerows(data)

def main():
    extract_values('search_value.sa.tok', 'pre_search_value.sa.tok')
    process_search_inputs('search_inputs.sa.tok', 'pre_search_inputs.sa.tok')
    process_search_files('pre_search_value.sa.tok', 'pre_search_inputs.sa.tok', 'EOS_search_inputs.sa.tok')
    pre_extract_EOS_pairs('EOS_search_inputs.sa.tok', 'extract_EOS_pairs.csv')

if __name__ == "__main__":
    main()
