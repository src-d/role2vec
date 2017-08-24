import os
import shutil

PROX_DIR = "/storage/timofei/role2vec/libs/ast2vec/uast_prox"
OUT_DIR = "/storage/timofei/role2vec/embeddings"


for fname in ["train", "valid", "test"]:
    with open("uasts_{}.txt".format(fname)) as fin:
        for line in fin:
            line = line.strip()
            uname = line[line.rfind("/") + 1:]
            letter = uname[len("uast_"):len("uast_") + 1]
            pname = os.path.join(PROX_DIR, letter, uname)
            if os.path.exists(pname):
                oname = os.path.join(OUT_DIR, "prox_{}".format(fname), letter)
                os.makedirs(oname, exist_ok=True)
                shutil.move(pname, oname)
