#!/usr/bin/python3

import sys
import yaml


class ChapterSplitter:
    """
        PnP document chapter splitter.
    """
    def __init__(self, config_file: str) -> None:
        self.config = self.read_config(config_file)
        self.content = dict()

    def read_config(self, config_file) -> dict:
        with open(config_file, 'r') as fr:
            config = yaml.load(fr, Loader=yaml.FullLoader)
        return config

    #   Chapter 1

    #   It is a truth universally acknowledged, that a single man in
    #   possession of a good fortune, must be in want of a wife.

    #   However little known the feelings or views of such a man may be
    #   on his first entering a neighbourhood, this truth is so well
    #   fixed in the minds of the surrounding families, that he is
    #   considered the rightful property of some one or other of their
    #   daughters.

    #   “My dear Mr. Bennet,” said his lady to him one day, “have you
    #   heard that Netherfield Park is let at last?”

    def parse_chapter(self) -> None:
        content_start_flag = False
        with open(self.config["DOC"], 'r', encoding="utf8") as fr:
            for line in fr:
                if line.startswith("      Chapter"):
                    if content_start_flag:
                        self.content[chap] = content
                    chap = line.strip().replace("Chapter", "Chap").replace(" ", "_")
                    content_start_flag = True
                    content = ""
                else:
                    if content_start_flag:
                        content += line.strip().replace('“', '"').replace('”', '"') + " "

    def write_result(self) -> None:
        for chap, content in self.content.items():
            with open(f"../data/{chap}.txt", 'w') as fw:
                fw.write(f"{content}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"#usage: python {sys.argv[0]} [config]")
        sys.exit()

    config_file = sys.argv[1]
    cs = ChapterSplitter(config_file)
    # print(cs.config)
    cs.parse_chapter()
    # print(cs.content["Chap_2"])
    cs.write_result()
