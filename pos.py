#!/usr/bin/python3
#coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import glob
import sys
import getopt
from datetime import datetime
import logging
import time

logging.getLogger("requests").setLevel(logging.WARNING)

# Suppress as many warnings as possible
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from ckiptagger import data_utils, construct_dictionary, WS, POS, NER


def main(argv):

    def print_word_pos_sentence(word_sentence, pos_sentence):
        assert len(word_sentence) == len(pos_sentence)
        for word, pos in zip(word_sentence, pos_sentence):
            print(f"{word}({pos})", end="\u3000")
        print()
        return

    def save_word_pos(word_sentence, pos_sentence, destfile):
        assert len(word_sentence) == len(pos_sentence)
        with open(destfile, "a") as out:
            for word, pos in zip(word_sentence, pos_sentence):
                out.write(f"{word}({pos})　")
            out.write("\n")
#                print(f"{word}({pos})", end="\u3000")
        print("save " + destfile + " done")
        return

#    inputfile = ''
#    outputfile = ''
#    try:
#        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
#    except getopt.GetoptError:
#        print ("test.py -i <inputfile> -o <outputfile>")
#        sys.exit(2)
#    for opt, arg in opts:
#        if opt == '-h':
#            print('test.py -i <inputfile> -o <outputfile>')
#            sys.exit()
#        elif opt in ("-i", "--ifile"):
#            inputfile = arg
#        elif opt in ("-o", "--ofile"):
#            outputfile = arg
#    print( 'Input file is "', inputfile)
#    print( 'Output file is "', outputfile)
#    return
    # Download data
    #data_utils.download_data("./")
    
    # Load model without GPU
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ws = WS("./data")
    pos = POS("./data")
    ner = NER("./data")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("above is loading model")
    
    # Load model with GPU
    # ws = WS("./data", disable_cuda=False)
    # pos = POS("./data", disable_cuda=False)
    # ner = NER("./data", disable_cuda=False)

    # Create custom dictionary
    word_to_weight = {
#        "土地公": 1,
        "土地婆": 1,
        "公有": 2,
        "": 1,
        "來亂的": "啦",
        "緯來體育台": 1,
        "風太": 1,
        "小乃": 1,
    }
    dictionary = construct_dictionary(word_to_weight)
    print(dictionary)
    
    # Run WS-POS-NER pipeline
    sentence_list = [
#        "傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。",
#        "美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。",
#        "小鈴、風太、美穗和另外一位同學一起走在校園裡。風太遞給小鈴一本書和一顆棒球。",
#        "",
#        "土地公有政策?？還是土地婆有政策。.",
#        "… 你確定嗎… 不要再騙了……",
#        "最多容納59,000個人,或5.9萬人,再多就不行了.這是環評的結論.",
#        "科長說:1,坪數對人數為1:3。2,可以再增加。",
    ]
    
    #read file
#    sentence_list = []
    while True:
        for file in os.listdir("./posinput"):
            if file.endswith(".txt"):
                with open(os.path.join("./posinput", file)) as my_file:
                    sentence_list = [line.rstrip() for line in my_file]

                if sentence_list[-1] != "%%EoF%%":
                    continue
                del sentence_list[-1]

#    with open(inputfile) as my_file:
#        sentence_list = [line.rstrip() for line in my_file]

#    word_sentence_list = ws(sentence_list)
#    word_sentence_list = ws(sentence_list, sentence_segmentation=True)
    # word_sentence_list = ws(sentence_list, recommend_dictionary=dictionary)
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                word_sentence_list = ws(sentence_list, coerce_dictionary=dictionary)
                pos_sentence_list = pos(word_sentence_list)
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print("above is pos tagging")
                entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print("above is entity")
    
                for i, sentence in enumerate(sentence_list):
                    print()
                    print(f"'{sentence}'")
#                print_word_pos_sentence(word_sentence_list[i],  pos_sentence_list[i])
                    save_word_pos(word_sentence_list[i],  pos_sentence_list[i], os.path.join("./posoutput", file))
                    print()
                    print("entities")
                    for entity in sorted(entity_sentence_list[i]):
                        print(entity)
                    print("================================")
                with open(os.path.join("./posoutput", file), "a") as out:
                    out.write("%%EoF%%")

            os.rename(os.path.join("./posinput", file), os.path.join("./backup", file))
        time.sleep(5)
    # Release model
    del ws
    del pos
    del ner
    
    # Show results
#    def print_word_pos_sentence(word_sentence, pos_sentence):
#        assert len(word_sentence) == len(pos_sentence)
#        for word, pos in zip(word_sentence, pos_sentence):
#            print(f"{word}({pos})", end="\u3000")
#        print()
#        return
    
#    for i, sentence in enumerate(sentence_list):
#        print()
#        print(f"'{sentence}'")
#        print_word_pos_sentence(word_sentence_list[i],  pos_sentence_list[i])
#        print()
#        print("entities")
#        for entity in sorted(entity_sentence_list[i]):
#            print(entity)
#        print("================================")
    return
    
if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit()
    


