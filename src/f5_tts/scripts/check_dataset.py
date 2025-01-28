import os.path

import click


@click.command
@click.option("--folder")
def check(folder):
    new_rows = []
    print("checking metadata")
    for row in open(os.path.join(folder, "metadata.csv"), 'r', encoding='utf-8').read().split("\n"):
        if "|" not in row:
            print("invalid data: " + row)
            continue
        file_name, text = row.split("|")
        if os.path.exists(os.path.join(folder, "wavs", file_name)):
            new_rows.append(row)
        else:
            print("wav file not exist: " + row)
    open(os.path.join(folder, "new_metadata.csv"), 'w', encoding='utf-8').write("\n".join(new_rows))
    file_names = [item.split("|")[0] for item in new_rows]
    print("checking wav file")
    for wav in os.listdir(os.path.join(folder, "wavs")):
        if wav not in file_names:
            print("invalid wav: " + wav)
            os.remove(os.path.join(folder, "wavs", wav))


if __name__ == '__main__':
    check()
