import argparse

parser = argparse.ArgumentParser(description='Generate file list for Emilia dataset')
parser.add_argument('--lang-code', type=str, default='EN', help='Language code (default: EN)')
parser.add_argument('--max-id', type=int, default=1139, help='Maximum ID number (default: 1139)')
parser.add_argument('--output-file', type=str, default='file_lists/en_yodas.txt', help='Output file path (default: file_lists/en_yodas.txt)')
args = parser.parse_args()

arr = []
for i in range(args.max_id + 1):
    arr.append(f"{args.lang_code}-B{i:06d}")

with open(args.output_file, "w") as f:
    f.write("\n".join(arr))