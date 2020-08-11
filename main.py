from model import *


def main():
    # Initialize GPT-2 Medium and generative class functions
    x = model()

    total_text = ""
    read_file = open("input.txt", "r")
    output_file = open("output.txt", "w")
    print("Begin typing")

    while True:
        file_body = read_file.read()
        cctd_string = total_text + file_body
        if (total_text == "") and (len(file_body) != 0):
            total_text = cctd_string
            output_file.write(total_text + x.generate_some_text(total_text, 1))

        if (len(cctd_string) > len(file_body)) and file_body != "":
            total_text = cctd_string
            output_file.write(
                total_text + x.generate_some_text(total_text, 1) + "\n")


if __name__ == "__main__":
    main()
