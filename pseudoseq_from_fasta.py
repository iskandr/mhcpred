import argparse

parser = argparse.ArgumentParser(
    description='Align MHC sequences')

parser.add_argument(
    "file",
    help="Directory which contains MHC sequence fasta files to align",
)

if __name__ == "__main__":
    args = parser.parse_args()
