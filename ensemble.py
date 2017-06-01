# Ensemble submission csv files together

import numpy as np
import pandas as pd
import itertools

fns = [ # files for ensembling
  "lstm.csv"
   "moe4_do.csv"
]
fn0, fn1 = fns # getting each file to variable

outfn="weighted_predictions.csv" # output file

def parse_line(ln):
  id_, vals = ln.strip().split(',') # split id and vals
  vals = vals.split() # split vals, which are space-separated
  ix = np.array(vals[::2]).astype(np.int16) # ix stores all predicted labels
  p = np.array(vals[1::2]).astype(np.float32) # p stores all predicted labels' probabilities
  df = pd.DataFrame(p, index=ix, columns=['p']) # puts into data frame
  return id_, df # returns the id and data frame

cnt = 0

with open(fn0,'r') as inf0, open(fn1, 'r') as inf1, open(outfn, 'w') as outf:
  first_line0 = next(inf0) # the header
  first_line1 = next(inf1)
  assert first_line0 == first_line1
  outf.write(first_line0) # writing the header
  stop = None
  for ln0, ln1 in itertools.islice(zip(inf0, inf1), stop):
    cnt += 1
    if cnt % 100 == 99:
        print(cnt)
    id0, df0 = parse_line(ln0)
    id1, df1 = parse_line(ln1)
    assert id0 == id1
    vid = id0
    df = pd.concat([df0, df1], axis=1).fillna(0)
    df.columns = ['p0','p1']
    df['wp'] = (df['p0'] + df['p1']) / 2.0
    df = df.sort_values('wp', ascending=False)
    df = df.iloc[:20,:]
    z = zip(list(df.index), list(df['wp']))
    out = " ".join(["%d %6f"%x for x in z])
    out = ",".join([vid, out])
    outf.write(out+'\n')