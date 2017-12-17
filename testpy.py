
#!/usr/bin/env python3

import glob

fname_list = glob.glob("./test/*.png")
for fname in sorted(fname_list):
    print(fname)
