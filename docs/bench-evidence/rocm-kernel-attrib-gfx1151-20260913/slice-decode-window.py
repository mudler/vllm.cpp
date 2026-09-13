import csv, sys, collections
p = sys.argv[1]
rows = [r for r in csv.DictReader(open(p, newline='')) if r.get("Kernel_Name")]
rows.sort(key=lambda r: int(r["Start_Timestamp"]))
print("total_rows", len(rows))
marks = [i for i, r in enumerate(rows) if "ArgmaxK" in r["Kernel_Name"]]
print("marks", len(marks), "rows_before_first_mark", marks[0] if marks else None)
segs = [(marks[i], marks[i+1]) for i in range(len(marks)-1)]
sizes = [b-a for a, b in segs]
print("hist", sorted(collections.Counter(sizes).items()))
odd = [(i, s) for i, s in enumerate(sizes) if s != collections.Counter(sizes).most_common(1)[0][0]]
print("deviating_indices", odd)
# longest contiguous equal-size stretch
runs=[]; cur_start=0
for i in range(1, len(sizes)+1):
    if i == len(sizes) or sizes[i] != sizes[cur_start]:
        runs.append((cur_start, i, sizes[cur_start])); cur_start = i
print("uniform_stretches", runs)
cand=[r for r in runs if r[1]-r[0] >= 50]
best = cand[-1] if cand else max(runs, key=lambda r: r[1]-r[0])
print("selected_stretch(last>=50)", best)
a, b, sz = best
lo, hi = marks[a], marks[b]      # [lo, hi] inclusive of the closing mark
out = sys.argv[2]
with open(out, 'w', newline='') as fh:
    wtr = csv.DictWriter(fh, fieldnames=rows[0].keys())
    wtr.writeheader()
    for r in rows[lo:hi+1]:
        wtr.writerow(r)
print("wrote", out, "rows", hi+1-lo, "steps", b-a)
