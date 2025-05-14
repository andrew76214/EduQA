import csv
from opencc import OpenCC

class SimplifiedChineseChecker:
    def __init__(self, input_csv):
        self.input_csv = input_csv
        self.converter = OpenCC('s2t')

    def find_simplified_characters(self, text):
        """Find Simplified Chinese characters in the given text."""
        converted_text = self.converter.convert(text)
        simplified_chars = [char for char, converted_char in zip(text, converted_text) if char != converted_char]
        return simplified_chars

    def check_csv_for_simplified_chinese(self):
        """Check the CSV file for Simplified Chinese characters."""
        with open(self.input_csv, mode='r', encoding='utf-8-sig') as f_in:
            reader = csv.DictReader(f_in)
            for idx, row in enumerate(reader, start=1):
                question = row.get("Question", "").strip()
                answer = row.get("Answer", "").strip()

                question_simplified = self.find_simplified_characters(question)
                answer_simplified = self.find_simplified_characters(answer)

                if question_simplified:
                    print(f"Row {idx}: Question contains Simplified Chinese characters: {''.join(question_simplified)}")
                if answer_simplified:
                    print(f"Row {idx}: Answer contains Simplified Chinese characters: {''.join(answer_simplified)}")

# Example usage
if __name__ == "__main__":
    checker = SimplifiedChineseChecker("../output_gpt4o_v1.1.csv")
    checker.check_csv_for_simplified_chinese()